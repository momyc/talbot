from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, partial, singledispatch, wraps
import asyncio
import logging
import os
import re

from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import FileChatMessageHistory
from langchain.vectorstores import Pinecone
from pydantic import BaseModel, BaseSettings
from telebot import types
from telebot.async_telebot import AsyncTeleBot
from telebot.util import quick_markup
import pinecone


class Settings(BaseSettings):

    telegram_bot_api_token: str
    default_model = 'gpt-3.5-turbo'

    pinecone_api_key: str
    pinecone_index: str
    pinecone_env: str

    openai_api_key: str
    huggingfacehub_api_token: str

    max_memory_slots: int = 5
    history_root: str = 'history'

    class Config:
        env_file = '.env'


SETTINGS = Settings()


class AsyncHuggingFaceHub(HuggingFaceHub):

    class Config:
        keep_untouched = ThreadPoolExecutor,

    executor = ThreadPoolExecutor()

    # Patch to make it "asynchronous"
    @staticmethod
    def make_async(executor):
        def decorator(func):
            @wraps(func)
            async def wrap(*args, **kwargs):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    executor, partial(func, *args, **kwargs))
            return wrap

        return decorator

    @make_async(executor)
    def arun(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @make_async(executor)
    def agenerate(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    @make_async(executor)
    def agenerate_prompt(self, *args, **kwargs):
        return self.generate_prompt(*args, **kwargs)

    @make_async(executor)
    def apredict(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @make_async(executor)
    def apredict_messages(self, *args, **kwargs):
        return self.predict_messages(*args, **kwargs)


def build_openai_llm(model):
    return ChatOpenAI(
        model_name=model,
        openai_api_key=SETTINGS.openai_api_key,
    )


def build_huggingface_llm(model):
    return AsyncHuggingFaceHub(
        repo_id=model,
        huggingfacehub_api_token=SETTINGS.huggingfacehub_api_token,
        task='text-generation',
    )


LLMS = {
    'OpenAI': {
        'factory': build_openai_llm,
        'models': [
            'gpt-3.5-turbo',
        ],
    },
    'HuggingFace': {
        'factory': build_huggingface_llm,
        'models': [
            'bigscience/bloom',
            'ElKulako/cryptobert',
            'TheBloke/WizardLM-7B-uncensored-GPTQ',
            'ehartford/WizardLM-13B-Uncensored',
            'mosaicml/mpt-7b-chat',
        ],
    },
}

MODELS_KEYBOARD = quick_markup(
    {
        f'{v} ({t})': {'callback_data': f'model-{v}'}
        for (t, d) in LLMS.items()
        for v in d['models']
    },
    row_width=1,
)


class ShitHappens:

    def handle(self, exc):
        logging.exception('''Something didn't work''')


pinecone.init(SETTINGS.pinecone_api_key, environment=SETTINGS.pinecone_env)
bot = AsyncTeleBot(
    SETTINGS.telegram_bot_api_token, exception_handler=ShitHappens())
chats = {}


@singledispatch
def get_chat(chat_id):
    try:
        chat = chats[chat_id]
    except KeyError:
        chat = Chat(id=chat_id, model=SETTINGS.default_model)
        chats[chat.id] = chat
        chat.chain
    return chat


@get_chat.register
def get_chat_for_message(msg: types.Message):
    return get_chat(msg.chat.id)


def command(*commands):
    def wrapper(func):
        return bot.message_handler(commands=commands)(func)
    return wrapper


def callback(pattern):
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    def wrapper(func):
        return bot.callback_query_handler(
            func=lambda call: pattern.match(call.data))

    return wrapper


@command('start')
async def on_start(msg):
    name = msg.from_user.first_name
    response = f'Hi {name}! Lets chat a little, shall we?'
    chat = get_chat(msg)
    await bot.send_message(chat.id, response)


@command('model')
async def on_model(msg):
    chat = get_chat(msg)
    await bot.send_message(
        chat.id, 'Please select a model:', reply_markup=MODELS_KEYBOARD)


@command('memory')
async def on_memory(msg):
    chat = get_chat(msg)
    slots_kbd = quick_markup(
        {
            chat.memory_head(100, i): {'callback_data': f'memory-{i}'}
            for i in range(SETTINGS.max_memory_slots)
        },
        row_width=1,
    )
    await bot.send_message(
        chat.id, 'Please select memory slot:',
        reply_markup=slots_kbd)


@command('info')
async def on_info(msg):
    chat = get_chat(msg)
    await bot.send_message(
        chat.id,
        f'Model: { chat.model }\n'
        f'Memory slot: { chat.memory_slot }\n'
        f'Your messages:\n'
        + '\n'.join(
            f'- {m.content}' for m in chat.memory.chat_memory.messages
            if m.type == 'human'))


@command('clear')
async def on_clear(msg):
    chat = get_chat(msg)
    chat.chain.memory.clear()
    await bot.send_message(chat.id, 'Conversation history is empty now :)')


@bot.message_handler(func=lambda msg: True)
async def on_message(msg):
    chat = get_chat(msg)
    await bot.send_message(chat.id, await chat.answer(msg))


@callback(r'^model-')
async def model_callback(call):
    chat = get_chat(call.message)
    model = call.data[6:]
    if model and model != chat.model:
        chat.model = model
        try:
            del chat.llm
        except AttributeError:
            pass

        await bot.delete_message(
            chat_id=chat.id, message_id=call.message.message_id)
        await bot.send_message(
            chat.id, f'Switched model to {chat.model}')


@callback(r'^memory-\d+')
async def memory_callback(call):
    chat = get_chat(call.message)
    slot = call.data[7:]
    if slot:
        chat.set_memory_slot(slot)
        await bot.delete_message(
            chat_id=chat.id, message_id=call.message.message_id)
        await bot.send_message(
            chat.id, f'Switched to conversation {chat.memory_slot}')


class Chat(BaseModel):

    id: int | str
    model: str
    memory_slot: int = 0
    history_root: str = SETTINGS.history_root

    class Config:
        keep_untouched = cached_property,

    @cached_property
    def llm(self):
        try:
            build_llm = next((
                d['factory'] for d in LLMS.values()
                if self.model in d['models']))
        except StopIteration:
            raise ValueError(f'Unknown model "{self.model}"')

        return build_llm(self.model)

    @cached_property
    def memory_folder(self):
        return f'{self.history_root}/{self.id:08x}'

    @cached_property
    def memory(self):
        folder = self.memory_folder
        os.makedirs(folder, exist_ok=True)

        chat_memory = FileChatMessageHistory(
            f'{self.memory_folder}/{self.memory_slot}.json')

        return ConversationBufferWindowMemory(
            chat_memory=chat_memory,
            memory_key='chat_history',
            return_messages=True,
            k=5)

    @cached_property
    def chain(self):
        embed = OpenAIEmbeddings(openai_api_key=SETTINGS.openai_api_key)

        vector_store = Pinecone(
            pinecone.Index(SETTINGS.pinecone_index),
            embed.embed_query,
            'text')

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            memory=self.memory,
            retriever=vector_store.as_retriever(),
        )

    async def answer(self, msg: types.Message):
        if msg.text:
            return await self.chain.arun(msg.text)

        return 'I can only respond to text messages'

    def set_memory_slot(self, slot):
        self.memory_slot = slot

        try:
            del self.memory
        except AttributeError:
            pass
        else:
            del self.chain

    def memory_head(self, max_len=50, slot=None):
        if slot is None or slot == self.memory_slot:
            messages = self.memory.chat_memory.messages
        else:
            messages = FileChatMessageHistory(
                f'{self.memory_folder}/{slot}.json').messages

        if messages:
            msg = messages[0].content
            if len(msg) > max_len:
                msg = msg[:max_len] + '...'
        else:
            msg = ''

        return msg
