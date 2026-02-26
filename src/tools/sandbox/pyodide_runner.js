/**
 * pyodide_runner.js — Lightweight Node.js script that executes Python code
 * inside a Pyodide (WebAssembly) sandbox.
 *
 * Usage:  echo 'print("hello")' | node pyodide_runner.js
 *
 * Output: JSON on stdout → { "stdout": "hello\n", "stderr": "", "exit_code": 0 }
 *
 * Requirements: npm package "pyodide" (auto-installed by the script on first run).
 */

async function main() {
    // Read Python code from stdin
    const chunks = [];
    for await (const chunk of process.stdin) {
        chunks.push(chunk);
    }
    const code = Buffer.concat(chunks).toString("utf-8");

    if (!code.trim()) {
        process.stdout.write(JSON.stringify({
            stdout: "",
            stderr: "No code provided.",
            exit_code: 1
        }));
        process.exit(0);
    }

    let loadPyodide;
    try {
        // Try to load pyodide — if not installed, exit with helpful message
        ({ loadPyodide } = require("pyodide"));
    } catch (_) {
        process.stdout.write(JSON.stringify({
            stdout: "",
            stderr: "Pyodide npm package not found. Run: npm install pyodide (in the project root or globally).",
            exit_code: 1
        }));
        process.exit(0);
    }

    let pyodide;
    try {
        pyodide = await loadPyodide({
            // Disable loading packages from the network for security
            indexURL: undefined,
        });
    } catch (err) {
        process.stdout.write(JSON.stringify({
            stdout: "",
            stderr: `Failed to initialize Pyodide: ${err.message}`,
            exit_code: 1
        }));
        process.exit(0);
    }

    // Capture stdout and stderr
    let capturedStdout = "";
    let capturedStderr = "";

    pyodide.setStdout({
        batched: (text) => { capturedStdout += text + "\n"; }
    });
    pyodide.setStderr({
        batched: (text) => { capturedStderr += text + "\n"; }
    });

    let exitCode = 0;
    try {
        await pyodide.runPythonAsync(code);
    } catch (err) {
        capturedStderr += err.message + "\n";
        exitCode = 1;
    }

    process.stdout.write(JSON.stringify({
        stdout: capturedStdout,
        stderr: capturedStderr,
        exit_code: exitCode
    }));
}

main().catch((err) => {
    process.stdout.write(JSON.stringify({
        stdout: "",
        stderr: `Runner error: ${err.message}`,
        exit_code: 1
    }));
    process.exit(0);
});
