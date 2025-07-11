const tf = require('@tensorflow/tfjs');
const natural = require('natural');
const math = require('mathjs');
const fs = require("fs");
const nlp = require('compromise');

// === Tokenizer ===
const tokenizer = new natural.TreebankWordTokenizer();

function tokenize(text) {
  return tokenizer.tokenize(text.toLowerCase());
}

// === Load database and build cache ===
const dbPath = "./db.json";
let db = [];
let normalizedMap = {};
try {
  db = JSON.parse(fs.readFileSync(dbPath));
  // Precompute normalized input map for fast lookup
  normalizedMap = Object.fromEntries(
    db.map(pair => [normalize(pair.input), pair])
  );
} catch {
  console.warn("No db.json found. Starting with empty DB.");
}

// === Dynamic Vocabulary ===
let vocabulary = Array.from(
  new Set(
    db.flatMap(pair => [
      ...tokenize(pair.input),
      ...(Array.isArray(pair.output)
        ? pair.output.flatMap(o => tokenize(o))
        : tokenize(pair.output))
    ])
  )
);

// === Vectorizer ===
const TfIdf = natural.TfIdf;
let tfidf = new TfIdf();

function buildTfidf() {
  tfidf = new TfIdf();
  db.forEach(pair => {
    tfidf.addDocument(pair.input.toLowerCase());
  });
}

function textToVector(text) {
  const tokens = tokenize(text);
  const vector = [];
  vocabulary.forEach(word => {
    vector.push(tfidf.tfidf(word, tfidf.documents.length - 1));
  });
  return vector;
}

// === Training Prep ===
let trainingData;
let model;
async function trainIfNeeded() {
  buildTfidf();
  if (!trainingData) {
    trainingData = db.map(pair => ({
      input: textToVector(pair.input),
      output: textToVector(
        Array.isArray(pair.output) ? pair.output.join(" ") : pair.output
      ),
    }));
  }
  if (!model) {
    // Define a deeper sequential model with multiple hidden layers and dropout
    model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [vocabulary.length], units: 64, activation: 'relu'}));
    model.add(tf.layers.dropout({rate: 0.3}));
    model.add(tf.layers.dense({units: 32, activation: 'relu'}));
    model.add(tf.layers.dropout({rate: 0.3}));
    model.add(tf.layers.dense({units: vocabulary.length, activation: 'sigmoid'}));
    model.compile({optimizer: 'adam', loss: 'binaryCrossentropy'});
    if (trainingData.length > 0) {
      const inputs = tf.tensor2d(trainingData.map(d => d.input));
      const outputs = tf.tensor2d(trainingData.map(d => d.output));
      await model.fit(inputs, outputs, {
        epochs: 200,
        batchSize: 8,
        validationSplit: 0.2,
        verbose: 0
      });
      inputs.dispose();
      outputs.dispose();
    }
  }
}
trainIfNeeded();

// === NLP-enhanced Matching ===
const wordnet = new natural.WordNet();

function normalize(text) {
  // Use compromise normalization for punctuation, whitespace, and case
  return nlp(text).normalize({punctuation:true, whitespace:true, case:true}).out('text');
}

function levenshtein(a, b) {
  return natural.LevenshteinDistance(a, b);
}

// === Synonym Expansion ===
const synonymMap = {
  tell: ["say", "share", "give", "show", "reveal", "inform", "communicate", "express"],
  fact: ["information", "detail", "data", "truth", "reality", "certainty"],
  about: ["regarding", "concerning", "on", "related to", "with respect to"],
  can: ["could", "would", "will", "may", "might", "shall"],
  please: ["kindly", "would you", "could you", "if you please", "do"],
  hello: ["hi", "hey", "greetings", "salutations"],
  bye: ["goodbye", "farewell", "see you", "later"],
  // Add more as needed
};

function expandSynonyms(tokens) {
  let expanded = new Set(tokens);
  tokens.forEach(token => {
    if (synonymMap[token]) {
      synonymMap[token].forEach(syn => expanded.add(syn));
    }
  });
  return Array.from(expanded);
}

function jaccardSimilarity(aTokens, bTokens) {
  const setA = new Set(aTokens);
  const setB = new Set(bTokens);
  const intersection = new Set([...setA].filter(x => setB.has(x)));
  const union = new Set([...setA, ...setB]);
  return intersection.size / union.size;
}

function bestMatch(inputText) {
  const normInput = normalize(inputText);
  const inputTokens = expandSynonyms(tokenize(normInput));
  let best = null;
  let bestScore = 0;
  let bestLev = Infinity;
  let bestOverlap = 0;
  db.forEach(pair => {
    const normDb = normalize(pair.input);
    const dbTokens = expandSynonyms(tokenize(normDb));
    // Jaccard similarity for paraphrase tolerance
    const score = jaccardSimilarity(inputTokens, dbTokens);
    // Levenshtein distance for typo-tolerance
    const lev = levenshtein(normInput, normDb);
    // Keyword overlap
    const overlap = inputTokens.filter(t => dbTokens.includes(t)).length;
    if (
      (score > bestScore) ||
      (score === bestScore && overlap > bestOverlap) ||
      (score === bestScore && overlap === bestOverlap && lev < bestLev)
    ) {
      bestScore = score;
      bestLev = lev;
      best = pair;
      bestOverlap = overlap;
    }
  });
  // Require a much higher similarity and overlap for a match
  if (bestScore > 0.6 && bestOverlap >= 3) return best;
  if (bestLev <= 1 && bestOverlap >= 3) return best;
  return null;
}

// === Generate Reply ===
async function generateResponse(inputText) {
  // Special sanity check for questions about the capital of planets or celestial bodies
  const planetList = [
    'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto',
    'sun', 'moon', 'io', 'europa', 'ganymede', 'callisto', 'titan', 'ceres', 'eris', 'haumea', 'makemake'
  ];
  const capitalMatch = inputText.match(/capital of ([a-z]+)/i);
  if (capitalMatch && planetList.includes(capitalMatch[1].toLowerCase())) {
    return `${capitalMatch[1][0].toUpperCase() + capitalMatch[1].slice(1)} is a planet, not a country, so technically it doesn't have a capital.`;
  }
  // Math problem detection
  const mathResult = trySolveMath(inputText);
  if (mathResult) return mathResult;
  // Fast exact/normalized match
  const normInput = normalize(inputText);
  let match = normalizedMap[normInput];
  if (!match) {
    match = bestMatch(inputText);
  }
  if (match) {
    if (Array.isArray(match.output)) {
      return match.output[Math.floor(Math.random() * match.output.length)];
    } else {
      return match.output;
    }
  }
  // Fallback: always return a clear message for unknowns
  return "I don't know the answer to that yet.";
}

// === Math Problem Detection & Solving ===
function trySolveMath(inputText) {
  // Use mathjs to evaluate math expressions
  try {
    // Replace words with operators for mathjs compatibility
    let expr = inputText.toLowerCase();
    expr = expr.replace(/plus|add(ed)? to/g, '+');
    expr = expr.replace(/minus|subtract(ed)?( from)?/g, '-');
    expr = expr.replace(/times|multipl(y|ied)? by|x/g, '*');
    expr = expr.replace(/divided by|over|÷/g, '/');
    expr = expr.replace(/to the power of|raised to|\^/g, '^');
    // Remove non-math characters except digits, operators, parentheses, decimal points, and spaces
    expr = expr.replace(/[^0-9+\-*/^(). ]/g, '');
    const result = math.evaluate(expr);
    if (result !== undefined) {
      return `Result: ${result}`;
    }
  } catch (e) {
    // Ignore errors and fallback to old method
  }
  return null;
}

// === Learn New Pair ===
function learn(input, output) {
  // If output is a string that looks like an array, parse it
  let outArr = output;
  if (typeof output === 'string') {
    try {
      const parsed = JSON.parse(output);
      if (Array.isArray(parsed)) outArr = parsed;
      else outArr = [output];
    } catch {
      outArr = [output];
    }
  }
  // Check if input already exists
  const normInput = normalize(input);
  const match = db.find(pair => normalize(pair.input) === normInput);
  if (match) {
    // Merge arrays, avoid duplicates
    const existing = Array.isArray(match.output) ? match.output : [match.output];
    match.output = Array.from(new Set([...existing, ...outArr]));
  } else {
    db.push({ input, output: outArr });
  }
  // Update normalizedMap for fast lookup
  normalizedMap[normInput] = db.find(pair => normalize(pair.input) === normInput);
  // Update vocab
  const newWords = [
    ...tokenize(input),
    ...outArr.flatMap(o => tokenize(o))
  ];
  newWords.forEach(w => {
    if (!vocabulary.includes(w)) vocabulary.push(w);
  });
  // Invalidate training cache
  trainingData = undefined;
  model = undefined;
  fs.writeFileSync(dbPath, JSON.stringify(db, null, 2));
  console.log("Learned:", input, "=>", outArr);
}

// === CLI Logic ===
const args = process.argv.slice(2);
if (args[0] === "learn") {
  // Support: node Tails_1.js learn '[{"input":"hi","output":["hello","hey"]},{"input":"bye","output":"goodbye"}]'
  if (args[1] && args[1].startsWith("[")) {
    try {
      const arr = JSON.parse(args[1]);
      if (Array.isArray(arr)) {
        arr.forEach(pair => {
          if (pair.input && pair.output) learn(pair.input, pair.output);
        });
        process.exit(0);
      }
    } catch (e) {
      console.log("Invalid multi-learn JSON array.");
      process.exit(1);
    }
  }
  // Single learn fallback
  const input = args[1];
  const output = args[2];
  if (input && output) {
    learn(input, output);
  } else {
    console.log("Usage: node Tails_1.js learn 'hello' 'hi'\n   or: node Tails_1.js learn '[{\\\"input\\\":\\\"hi\\\",\\\"output\\\":[\\\"hello\\\"]}]'");
  }
} else if (args.length > 0) {
  const inputText = args.join(" ");
  generateResponse(inputText).then(response => {
    console.log("AI:", response || "(I don't know yet)");
  });
} else {
  console.log("Usage:\n  node Tails_1.js learn 'hi' 'hello'\n  node Tails_1.js 'your input'");
}
