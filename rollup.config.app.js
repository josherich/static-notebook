const definition = require("./package.json");
const dependencies = Object.keys(definition.dependencies);
import babel from 'rollup-plugin-babel';

export default {
  input: "app.js",
  external: dependencies,
  output: {
    extend: true,
    file: "dist/app.js",
    format: "umd",
    globals: {
      "markdown-it": "markdownit",
      "@shopify/draggable": "draggable",
      "turndown": "TurndownService"
    },
    plugins: [
      babel({
        exclude: 'node_modules/**' // only transpile our source code
      })
    ],
    name: "sd"
  }
};