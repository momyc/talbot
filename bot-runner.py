#!/bin/env python

import asyncio

from talbot.factory import bot


asyncio.run(bot.infinity_polling())
