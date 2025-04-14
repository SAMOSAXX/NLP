#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 18:24:03 2025

@author: aayushsrivatsav
"""

from openai import OpenAI
api_key = "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"
client = OpenAI(api_key=api_key)

# Get user input
user_message = input("Enter your message: ")

try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )
        print(response.choices[0].message.content)
except Exception as e:
        print(f"Error in semantic comparison: {e}")