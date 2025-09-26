var path = require("path");
// To build use `webpack --mode production` then copy bundle.js to resources

var buildDir = path.resolve(__dirname, "build");

var moduleConfig = {
  rules: [
    {
      test: /\.css$/,
      use: ["style-loader", "css-loader"]
    },
    {
      test: /\.js[x]?$/,
      exclude: /(node_modules)/,
      loader: "babel-loader",
      options: {
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
      test_bundle: "./test_bundle.js"
    },
    output: {
      path: buildDir,
      filename: "[name].js"
    },
    module: moduleConfig,
    resolve: resolveConfig,
    devServer: {
      static: buildDir
    }
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
