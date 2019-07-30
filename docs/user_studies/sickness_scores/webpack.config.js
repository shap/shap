var path = require("path");

var buildDir = path.resolve(__dirname, 'static/js');
module.exports = {
  entry: {bundle: "./index.jsx"},
  output: {
    path: buildDir,
    filename: "[name].js"
  },

  module: {
    loaders: [
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
