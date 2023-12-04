import React from 'react';
import {Link, hashHistory} from 'react-router';
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider';
import RaisedButton from 'material-ui/RaisedButton';
import TextField from 'material-ui/TextField';

export default class QuestionA1 extends React.Component {
  constructor() {
    super();
    this.state = {
      fever: 0,
      cough: 0,
      headache: 0,
      person: 0,
      score: 0
    };
    this.last_time = new Date().getTime();
    this.handleChange = this.handleChange.bind(this);

    this.symptoms = [
      {
        'fever': true,
        'cough': true,
        'headache': true
      },
      {
        'fever': false,
        'cough': true,
        'headache': true
      },
      {
        'fever': false,
        'cough': false,
        'headache': true
      }
    ]

    this.score_names = ["A", "B", "C", "D"];
    this.score_methods = [
      { // XOR
        "true true": 4,
        "false true": 8,
        "false false": 0
      },
      { // AND
        "true true": 10,
        "false true": 2,
        "false false": 0
      },
      { // OR
        "true true": 10,
        "false true": 8,
        "false false": 0
      },
      { // SUM
        "true true": 4,
        "false true": 2,
        "false false": 0
      }
    ];


    window.psiTurk.recordTrialData({
      'mark': "question"+this.score_names[this.state.score]+this.person+"_start",
      'time': this.last_time.valueOf()
    });
  }

  handleChange(event) {
    if (event.target.value !== undefined) {
      var data = {};
      data[event.target.name] = event.target.value;
      this.setState(data);
    }
  }

  saveAnswers() {
    if (this.state.person != 2 && parseInt(this.state.fever) + parseInt(this.state.cough) + parseInt(this.state.headache) === 0) {
      alert("Please consider carefully how to assign blame among the symptoms before submitting.");
    } else {
      window.psiTurk.recordTrialData({
        'mark': "question"+this.score_names[this.state.score]+this.state.person+"_stop",
        'fever': this.symptoms[this.state.person]['fever'],
        'cough': this.symptoms[this.state.person]['cough'],
        'headache': this.symptoms[this.state.person]['headache'],
        'fever_credit': parseInt(this.state.fever),
        'cough_credit': parseInt(this.state.cough),
        'headache_credit': parseInt(this.state.headache),
        'time': new Date().valueOf(),
        'condition': condition,
        'response_time': new Date().getTime() - this.last_time
      });

      this.state.fever = 0;
      this.state.cough = 0;
      this.state.headache = 0;

      if (this.state.person < 2) {
        this.state.person++;
      } else if (this.state.score < 3) {
        this.state.person = 0;
        this.state.score++;
      } else {
        hashHistory.push("/debrief");
      }
      this.setState(this.state);
      this.last_time = new Date().getTime();
    }
  }

  render() {

    var fever = this.symptoms[this.state.person]['fever'];
    var cough = this.symptoms[this.state.person]['cough'];

    var interaction_effect_rule;
    var interaction_effect_result;
    if (this.state.score == 0) {
      interaction_effect_rule = (
        <tr>
          <td style={{textAlign: "right", padding: "3px"}}>
            If they have a fever <i>OR</i> a cough but not both:
          </td>
          <td>
            +6 points
          </td>
        </tr>
      );

      if ((fever || cough) && !(fever && cough)) {
        interaction_effect_result = (
          <tr>
            <td style={{textAlign: "right", padding: "3px"}}>
              They have a fever <i>OR</i> a cough but not both:
            </td>
            <td>
              <b>+6 points</b>
            </td>
          </tr>
        );
      }
    } else if (this.state.score == 1) {
      interaction_effect_rule = (
        <tr>
          <td style={{textAlign: "right", padding: "3px"}}>
            If they have a fever <i>AND</i> a cough:
          </td>
          <td>
            +6 points
          </td>
        </tr>
      );

      if (fever && cough) {
        interaction_effect_result = (
          <tr>
            <td style={{textAlign: "right", padding: "3px"}}>
              They have a fever <i>AND</i> a cough:
            </td>
            <td>
              <b>+6 points</b>
            </td>
          </tr>
        );
      }
    } else if (this.state.score == 2) {
      interaction_effect_rule = (
        <tr>
          <td style={{textAlign: "right", padding: "3px"}}>
            If they have a fever <i>OR</i> a cough:
          </td>
          <td>
            +6 points
          </td>
        </tr>
      );

      if (fever || cough) {
        interaction_effect_result = (
          <tr>
            <td style={{textAlign: "right", padding: "3px"}}>
              They have a fever <i>OR</i> a cough:
            </td>
            <td>
              <b>+6 points</b>
            </td>
          </tr>
        );
      }
    }

    var fever_effect_result;
    if (fever) {
      fever_effect_result = (
        <tr>
          <td style={{textAlign: "right", padding: "3px"}}>
            They have a fever:
          </td>
          <td>
            <b>+2 points</b>
          </td>
        </tr>
      );
    }
    var cough_effect_result;
    if (cough) {
      cough_effect_result = (
        <tr>
          <td style={{textAlign: "right", padding: "3px"}}>
            They have a cough:
          </td>
          <td>
            <b>+2 points</b>
          </td>
        </tr>
      );
    }

    if (!fever && !cough) {
      fever_effect_result = (
        <tr>
          <td style={{textAlign: "right", padding: "3px"}}>
            They have neither a fever nor a cough.
          </td>
          <td>

          </td>
        </tr>
      );
    }

    var total_score = this.score_methods[this.state.score][fever + " " + cough];

    return (
      <MuiThemeProvider>
        <div id="container-instructions">

        	<h2>Assign blame among symptoms (score {this.score_names[this.state.score]}, person {this.state.person+1})</h2>

        	<hr/>

        	<div className="instructions well">

              <p>"<b>Sickness score {this.score_names[this.state.score]}</b>" is computed in the following manner:</p>

              <div style={{textAlign: "center"}}>
              <table style={{display: "inline-block", textAlign: "left"}}><tbody>
                <tr>
                  <td style={{textAlign: "right", padding: "3px"}}>
                    If they have a fever:
                  </td>
                  <td>
                    +2 points
                  </td>
                </tr>
                <tr>
                  <td style={{textAlign: "right", padding: "3px"}}>
                    If they have a cough:
                  </td>
                  <td>
                    +2 points
                  </td>
                </tr>
                {interaction_effect_rule}
              </tbody></table>
              </div>
              <br/>
              <b>Person {this.state.person+1}</b> has the following symptoms:<br/>
              <div style={{textAlign: "center", marginTop: "-20px"}}>
              <table style={{display: "inline-block", textAlign: "left"}}><tbody>
                <tr>
                  <td style={{textAlign: "right", padding: "3px"}}>
                    Fever:
                  </td>
                  <td style={{fontWeight: "bold", color: "#990000"}}>
                    {this.symptoms[this.state.person]['fever'] ? 'YES' : 'NO'}
                  </td>
                </tr>
                <tr>
                  <td style={{textAlign: "right", padding: "3px"}}>
                    Cough:
                  </td>
                  <td style={{fontWeight: "bold", color: "#990000"}}>
                  {this.symptoms[this.state.person]['cough'] ? 'YES' : 'NO'}
                  </td>
                </tr>
                <tr>
                  <td style={{textAlign: "right", padding: "3px"}}>
                    Headache:
                  </td>
                  <td style={{fontWeight: "bold", color: "#990000"}}>
                  {this.symptoms[this.state.person]['headache'] ? 'YES' : 'NO'}
                  </td>
                </tr>
              </tbody></table>
              </div>
              This leads to a total sickness score {this.score_names[this.state.score]} value of <b>{total_score} points</b> because:<br/>
              <div style={{textAlign: "center"}}>
              <table style={{display: "inline-block", textAlign: "left"}}><tbody>
                {fever_effect_result}
                {cough_effect_result}
                {interaction_effect_result}
              </tbody></table>
              </div>
              <br/>
              Using numbers please assign blame for their sickness score of <b>{total_score} points</b> among the following symptoms{this.state.person == 2 ? ' (zero blame is okay)' : ''}:<br/>

            <br/>
              <div style={{textAlign: "center"}}>
              <table style={{display: "inline-block", textAlign: "left"}}><tbody>
                <tr>
                  <td style={{textAlign: "center", width: "150px"}}>
                    <TextField name="fever" value={this.state.fever} onChange={this.handleChange} style={{width: "20px", marginLeft: 3, marginRight: 3}} /> point(s)<br/>
                    Fever: <b style={{color: "#990000"}}>{this.symptoms[this.state.person]['fever'] ? 'YES' : 'NO'}</b>
                  </td>
                  <td style={{textAlign: "center", width: "150px"}}>
                    <TextField name="cough" value={this.state.cough} onChange={this.handleChange} style={{width: "20px", marginLeft: 3, marginRight: 3}} /> point(s)<br/>
                    Cough: <b style={{color: "#990000"}}>{this.symptoms[this.state.person]['cough'] ? 'YES' : 'NO'}</b>
                  </td>
                  <td style={{textAlign: "center", width: "150px"}}>
                    <TextField name="headache" value={this.state.headache} onChange={this.handleChange} style={{width: "20px", marginLeft: 3}} /> point(s)<br/>
                    Headache: <b style={{color: "#990000"}}>{this.symptoms[this.state.person]['headache'] ? 'YES' : 'NO'}</b>
                  </td>
                </tr>
              </tbody></table>
              </div>

        	</div>
          <div style={{textAlign: "center"}}>
          <RaisedButton label={"Submit answer for score " + this.score_names[this.state.score] + ", person " + (this.state.person+1)} primary={true} onClick={()=>this.saveAnswers()} />
          </div>

        </div>
      </MuiThemeProvider>
    );
  }
}
