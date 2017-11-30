import random
import json

fallback = ['Sorry, could you say that again?', 'Sorry, can you say that again?', 
            'Can you say that again?', "Sorry, I didn't get that.", 
            'Sorry, what was that?', 'One more time?', 'What was that?', 'Say that again?',
            "I didn't get that.", "Please rephrase your question.", 
            "Please check the spelling."]

# =============================================================================
# module to handle small talk and fall back
# =============================================================================

class SmallTalk:
    def __init__(self, spacy_model):
        with open('path/to/your/small_talk_json_file') as data_file:
            self.smalltalk = json.load(data_file)
            self.nlp = spacy_model

    def get_reply(self, query):
        score = 0.0
        reply = None
        for key, values in self.smalltalk.items():
            score = max(score,
                        max([self.similarity(query, val) for val in values[0]]))
            for val in values[0]:
                if self.similarity(query, val) == score:
                    reply = random.choice(values[1]) if isinstance(values[1], list) else values[1]

        return reply if score > 0.7 else random.choice(fallback)

    def similarity(self, query1, query2):
        query1 = query1.lower()
        query2 = query2.lower()
        return self.nlp(query1).similarity(self.nlp(query2))
