module.exports = {
  moduleNameMapper: {
    // Fix Jest import of D3. See https://github.com/jestjs/jest/issues/12036
    '^d3$': '<rootDir>/node_modules/d3/dist/d3.min.js',
    "^d3-(.*)$": `<rootDir>/node_modules/d3-$1/dist/d3-$1`,
  },
  testEnvironment: 'jsdom'
};
