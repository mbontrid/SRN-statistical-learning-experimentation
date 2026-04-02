import json
import random
import string


class PatternStringGenerator:
    def __init__(self, pattern, alphabet=string.ascii_lowercase, reserved=None):
        self.pattern = pattern
        self.alphabet = list(alphabet)
        self.reserved = set(reserved) if reserved else set()

        self.usable_alphabet = [c for c in self.alphabet if c not in self.reserved]
        self.tokens = self._parse_pattern()

    # -------------------------
    # Public API
    # -------------------------

    def generate(self, n=1, unique=False):
        results = set() if unique else []

        while True:
            s = self._generate_one()

            if unique:
                results.add(s)
                if len(results) >= n:
                    return list(results)
            else:
                results.append(s)
                if len(results) >= n:
                    return results

    def generate_one(self):
        return self._generate_one()
    
    def sequences_to_json(self, sequences, filename):
        extracted_sequences = []
        for sequence in sequences:
            sequence_string = [letter for letter in sequence]
            extracted_sequences.append(sequence_string)
        with open(f'{filename}.json', 'w') as fp:
            json.dump({'extracted_sequences': extracted_sequences}, fp)


    # -------------------------
    # Generation
    # -------------------------

    def _generate_one(self):
        generated = []

        for token in self.tokens:
            if token[0] == "repeat":
                _, subtoken, min_n, max_n = token
                count = random.randint(min_n, max_n)

                for _ in range(count):
                    generated.append(self._generate_char(subtoken))
            else:
                generated.append(self._generate_char(token))

        return "".join(generated)

    def _generate_char(self, token):
        token_type, value = token

        if token_type == "fixed":
            return value

        elif token_type == "set":
            return random.choice(value)

        elif token_type == "var":
            p = value
            k = max(1, int(len(self.usable_alphabet) * p / 100))
            subset = random.sample(self.usable_alphabet, k)
            return random.choice(subset)

        else:
            raise ValueError(f"Unknown token type: {token_type}")

    # -------------------------
    # Parsing
    # -------------------------

    def _parse_pattern(self):
        tokens = []
        i = 0
        pattern = self.pattern

        while i < len(pattern):
            char = pattern[i]

            #Character set [ ... ]
            if char == "[":
                j = pattern.find("]", i)
                if j == -1:
                    raise ValueError("Unclosed '['")

                content = pattern[i + 1:j]
                expanded = self._expand_range(content)
                token = ("set", expanded)
                i = j + 1

            #Wildcard
            elif char == ".":
                token = ("set", self.usable_alphabet)
                i += 1

            #Variability %p
            elif char == "%":
                j = i + 1
                while j < len(pattern) and pattern[j].isdigit():
                    j += 1

                p = int(pattern[i + 1:j])
                token = ("var", p)
                i = j

            #Fixed character
            else:
                token = ("fixed", char)
                i += 1

            #Repetition
            if i < len(pattern) and pattern[i] == "{":
                j = pattern.find("}", i)
                if j == -1:
                    raise ValueError("Unclosed '{'")

                content = pattern[i + 1:j]

                if "," in content:
                    parts = content.split(",")
                    min_n = int(parts[0]) if parts[0] else 0
                    max_n = int(parts[1]) if parts[1] else min_n
                else:
                    min_n = max_n = int(content)

                tokens.append(("repeat", token, min_n, max_n))
                i = j + 1
            else:
                tokens.append(token)

        return tokens

    def _expand_range(self, content):
        result = []
        i = 0

        while i < len(content):
            if i + 2 < len(content) and content[i + 1] == "-":
                start = content[i]
                end = content[i + 2]
                result.extend(chr(c) for c in range(ord(start), ord(end) + 1))
                i += 3
            else:
                result.append(content[i])
                i += 1

        return result

    

if __name__ == '__main__':
    gen = PatternStringGenerator(
        pattern="A[A-Z]E.{2,4}C%50.",
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        reserved={"A", "B", "C", "D"}
    )
    
    samples = gen.generate(n=1000)
    gen.sequences_to_json(samples, 'test')
    
