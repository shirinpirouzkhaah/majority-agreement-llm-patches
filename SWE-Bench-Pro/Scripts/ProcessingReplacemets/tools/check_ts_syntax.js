const ts = require("/opt/homebrew/lib/node_modules/typescript/lib/typescript.js");

const filePath = process.argv[2];

if (!filePath) {
  console.error("Usage: node check_ts_syntax.js <file>");
  process.exit(2);
}

const fs = require("fs");
const sourceText = fs.readFileSync(filePath, "utf8");

const sourceFile = ts.createSourceFile(
  filePath,
  sourceText,
  ts.ScriptTarget.Latest,
  true
);

const diagnostics = sourceFile.parseDiagnostics || [];

if (diagnostics.length === 0) {
  process.exit(0);
}

for (const diagnostic of diagnostics) {
  const pos = diagnostic.file.getLineAndCharacterOfPosition(diagnostic.start);
  const message = ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n");
  console.error(`${filePath}:${pos.line + 1}:${pos.character + 1} - ${message}`);
}

process.exit(1);
