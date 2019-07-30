import React from 'react'
import {Router, Route, hashHistory} from 'react-router'
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'
import Instructions from './Instructions'
import Questions from './Questions'
import Debrief from './Debrief'

export default class Main extends React.Component {
  constructor() {
    super();
  }

  render() {
    return (
      <MuiThemeProvider>
        <div>
          <Router history={hashHistory}>
            <Route path="/" component={Instructions} />
            <Route path="/questions" component={Questions} />
            <Route path="/debrief" component={Debrief} />
          </Router>
        </div>
      </MuiThemeProvider>
    );
  }
}
