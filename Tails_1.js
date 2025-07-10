const brain = require("brain.js");
const fs = require("fs");
const nlp = require('compromise');

// === Tokenizer ===
function tokenize(text) {
  return text.toLowerCase().split(/\s+/).filter(Boolean);
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
function textToVector(text) {
  const tokens = tokenize(text);
  return vocabulary.map(word => (tokens.includes(word) ? 1 : 0));
}

// === Training Prep ===
let trainingData;
let net;
function trainIfNeeded() {
  if (!trainingData) {
    trainingData = db.map(pair => ({
      input: textToVector(pair.input),
      output: textToVector(
        Array.isArray(pair.output) ? pair.output.join(" ") : pair.output
      ),
    }));
  }
  if (!net) {
    net = new brain.NeuralNetwork({ hiddenLayers: [8] });
    if (trainingData.length > 0) {
      net.train(trainingData, {
        log: false,
        iterations: 1000,
        errorThresh: 0.01,
      });
    }
  }
}
trainIfNeeded();

// === NLP-enhanced Matching ===
function normalize(text) {
  // Lowercase, remove punctuation, and lemmatize
  return nlp(text).normalize({punctuation:true, whitespace:true, case:true}).out('text');
}

function levenshtein(a, b) {
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  const matrix = [];
  for (let i = 0; i <= b.length; i++) matrix[i] = [i];
  for (let j = 0; j <= a.length; j++) matrix[0][j] = j;
  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }
  return matrix[b.length][a.length];
}

function bestMatch(inputText) {
  const normInput = normalize(inputText);
  let best = null;
  let bestScore = 0;
  let bestLev = Infinity;
  db.forEach(pair => {
    const normDb = normalize(pair.input);
    // Simple similarity: count shared words
    const inputWords = new Set(normInput.split(' '));
    const dbWords = new Set(normDb.split(' '));
    const shared = [...inputWords].filter(w => dbWords.has(w)).length;
    const score = shared / Math.max(inputWords.size, dbWords.size);
    // Levenshtein distance for typo-tolerance
    const lev = levenshtein(normInput, normDb);
    if (score > bestScore || (score === bestScore && lev < bestLev)) {
      bestScore = score;
      bestLev = lev;
      best = pair;
    }
  });
  // Accept if similarity is reasonable or Levenshtein is very close
  if (bestScore > 0.25 || bestLev <= 2) return best;
  return null;
}

// === Generate Reply ===
function generateResponse(inputText) {
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
  // Fallback to neural net
  trainIfNeeded();
  const inputVec = textToVector(inputText);
  const outputVec = net.run(inputVec);
  const tokens = outputVec
    .map((value, i) => ({ word: vocabulary[i], value }))
    .filter(obj => obj.value > 0.3)
    .sort((a, b) => b.value - a.value)
    .map(obj => obj.word);
  return tokens.join(" ");
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
  net = undefined;
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
  const response = generateResponse(inputText);
  console.log("AI:", response || "(I don't know yet)");
} else {
  console.log("Usage:\n  node Tails_1.js learn 'hi' 'hello'\n  node Tails_1.js 'your input'");
}
