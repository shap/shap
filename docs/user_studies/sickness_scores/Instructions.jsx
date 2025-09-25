import React from 'react';
import {hashHistory} from 'react-router';
import RaisedButton from 'material-ui/RaisedButton';

export default class Instructions extends React.Component {
  constructor() {
    super();
    this.next_page = this.next_page.bind(this);

    this.start_time = new Date().getTime();
    window.psiTurk.recordTrialData({
      'mark': "instructions_start",
      'condition': condition,
      'time': this.start_time.valueOf()
    });
  }

  next_page() {
    window.psiTurk.recordTrialData({
      'mark': "instructions_stop",
      'time': new Date().valueOf(),
      'condition': condition,
      'response_time': new Date() - this.start_time
    });

    hashHistory.push("/questions");
    // if (condition === 0) hashHistory.push("/allocate1");
    // else if (condition === 1) hashHistory.push("/allocate2");
  }

  render() {
    return (
      <div id="container-instructions">
      	<h1>Instructions</h1>
      	<hr/>

      	<div className="instructions well">
          You will be asked to allocate blame for a person's "sickness score",
          where 0 is healthy and higher numbers mean more sickness. There are
          <i> 4 different ways</i> to measure sickness (A, B, C and D). There
          are also <i>3 different people</i> for each score. This leads to 12
          total questions. Each question should not take very long to
          complete, the goal is simply for you to assign blame among a person's
          symptoms in a way you consider the most intuitive.
      	</div>

        <div style={{textAlign: "center"}}>
          <RaisedButton label="Next >" primary={true} onClick={this.next_page} />
        </div>
      </div>
    );
  }
}
