const definition = require("./package.json");
const dependencies = Object.keys(definition.dependencies);
import babel from 'rollup-plugin-babel';

export default {
  input: "index",
  external: dependencies,
  output: {
    extend: true,
    file: `dist/${definition.name}.js`,
    format: "umd",
    globals: {
      'markdown-it': 'markdownit'
    },
    plugins: [
      babel({
        exclude: 'node_modules/**' // only transpile our source code
      })
    ],
    name: "sd"
  }
};