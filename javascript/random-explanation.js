import { range } from "lodash";

let seed = 1;
function random() {
  var x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
}

export default function() {
  return {
    outNames: ["Probability of flower"],
    baseValue: 0.2,
    link: "identity",
    features: [
      { name: "F1", effect: 0.0, value: 1 },
      { name: "F2", effect: -0.6, value: 1 },
      { name: "F3", effect: -0.2, value: 2 },
      { name: "F4", effect: 0, value: 0 }
    ]
    // range(20).map(i => ({
    //   name: 'value_'+i,
    //   effect: random()-0.5
    // }))
  };
}
