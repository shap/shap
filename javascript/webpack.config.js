var path = require("path");
// To build use `webpack -p` then copy bundle.js to resources

var buildDir = path.resolve(__dirname, 'build');
module.exports = {
  entry: {bundle: "./index.jsx", test_bundle: "./test.js"},
  output: {
    path: buildDir,
    filename: "[name].js"
  },

  module: {
    rules: [
      {
        test: /\.css$/, loader: "style!css"
      },
      {
        test: /\.js[x]?$/,
        exclude: /(node_modules|bower_components)/,
        loader: 'babel-loader',
        query: {
          presets: ['es2015', 'react']
        }
      }
    ]
  },

  resolve: {
		extensions: ['*', '.js', '.jsx', '.json']
	}
};
