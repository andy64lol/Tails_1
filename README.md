# Tails_1

A simple NLP AI model using brain.js.

## Description

Tails_1.js is a lightweight natural language processing AI built with [brain.js](https://github.com/BrainJS/brain.js) and [compromise](https://github.com/spencermountain/compromise). It uses a dynamic vocabulary and vectorization to train a neural network on input-output pairs stored in a JSON database. The AI supports fuzzy matching with Levenshtein distance for typo tolerance and can learn new input-output pairs dynamically.

## Installation

1. Clone the repository or download the files.
2. Install dependencies using npm:

```bash
npm install
```

## Usage

### Running the AI

To run the AI and get a response for an input, use:

```bash
node Tails_1.js "your input text"
```

Example:

```bash
node Tails_1.js "hello"
```

### Learning New Input-Output Pairs

You can teach the AI new responses using the `learn` command.

- To learn a single pair:

```bash
node Tails_1.js learn "input text" "output text"
```

- To learn multiple pairs at once, provide a JSON array:

```bash
node Tails_1.js learn '[{"input":"hi","output":["hello","hey"]},{"input":"bye","output":"goodbye"}]'
```

## Features

- **NLP with brain.js and compromise:** Combines neural networks with natural language processing for better understanding.
- **Dynamic Vocabulary and Vectorization:** Automatically builds vocabulary from the database and converts text to vectors for training.
- **Fuzzy Matching:** Uses Levenshtein distance and shared word similarity to find the best matching response even with typos or variations.
- **Persistent Learning:** Stores learned pairs in `db.json` for future use.

## Author

andy64lol

## License

This project is licensed under the MIT License.
