var path = require("path");
// To build use `webpack -p` then copy bundle.js to resources

var buildDir = path.resolve(__dirname, "build");

var moduleConfig = {
  rules: [
    {
      test: /\.css$/,
      loader: "style!css"
    },
    {
      test: /\.js[x]?$/,
      exclude: /(node_modules)/,
      loader: "babel-loader",
      query: {
        presets: ["@babel/preset-env", "@babel/preset-react"]
      }
    }
  ]
};

var resolveConfig = {
  extensions: ["*", ".js", ".jsx", ".json"]
};
module.exports = [
  {
    entry: {
      bundle: "./index.jsx",
      test_bundle: "./test.js"
    },
    output: {
      path: buildDir,
      filename: "[name].js"
    },
    module: moduleConfig,
    resolve: resolveConfig
  },
  {
    entry: {
      index: "./visualizers/index.jsx"
    },
    output: {
      path: buildDir,
      filename: "[name].js",
      libraryTarget: "commonjs2"
    },
    module: moduleConfig,
    resolve: resolveConfig
  }
];
