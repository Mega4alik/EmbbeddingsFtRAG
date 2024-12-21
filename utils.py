# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import codecs
import io
import re
import json
import numpy as np
from scipy import spatial


class myOpenAI():
    def __init__(self):    
        from openai import OpenAI
        OPENAI_API_KEY = ""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.embedding_name = "text-embedding-3-large"
        
    def get_embedding(self, text):       
       return self.client.embeddings.create(input = [text], model=self.embedding_name).data[0].embedding


def file_put_contents(filename, st):
    file = codecs.open(filename, "w", "utf-8")
    file.write(st)
    file.close()

def file_get_contents(name):
    f = io.open(name, mode="r", encoding="utf-8") #utf-8 | Windows-1252
    return f.read()

def timediff(time1, time2):
    return int( (time2 - time1) * 1000 ) #ms

def remove_multiple_spaces(text):
    for _ in range(2): text = text.replace("  ", " ")    
    return text

def cosine_similarity(v1, v2):
  return 1 - spatial.distance.cosine(v1, v2)