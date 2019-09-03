import React from "react";
import { scaleLinear } from "d3-scale";
import { format } from "d3-format";
import { sortBy, reverse, max, map } from "lodash";
import colors from "../color-set";

class SimpleListVisualizer extends React.Component {
  constructor() {
    super();
    this.width = 100;
    window.lastSimpleListInstance = this;
    this.effectFormat = format(".2");
  }

  render() {
    //Verify custom color map
    let plot_colors=undefined;
    if (typeof this.props.plot_cmap === "string")
    {
      if (!(this.props.plot_cmap in colors.colors))
      {
        console.log("Invalid color map name, reverting to default.");
        plot_colors=colors.colors.RdBu;
      }
      else
      {
        plot_colors = colors.colors[this.props.plot_cmap]
      }
    }
    else if (Array.isArray(this.props.plot_cmap)){
      plot_colors = this.props.plot_cmap
    }

    console.log(this.props.features, this.props.features);
    this.scale = scaleLinear()
      .domain([0, max(map(this.props.features, x => Math.abs(x.effect)))])
      .range([0, this.width]);

    // build the rows of the plot
    let sortedFeatureInds = reverse(
      sortBy(Object.keys(this.props.features), k =>
        Math.abs(this.props.features[k].effect)
      )
    );
    let rows = sortedFeatureInds.map(k => {
      let x = this.props.features[k];
      let name = this.props.featureNames[k];
      let style = {
        width: this.scale(Math.abs(x.effect)),
        height: "20px",
        background:
          x.effect < 0
            ? plot_colors[0]
            : plot_colors[1],
        display: "inline-block"
      };
      let beforeLabel;
      let afterLabel;
      let beforeLabelStyle = {
        lineHeight: "20px",
        display: "inline-block",
        width: this.width + 40,
        verticalAlign: "top",
        marginRight: "5px",
        textAlign: "right"
      };
      let afterLabelStyle = {
        lineHeight: "20px",
        display: "inline-block",
        width: this.width + 40,
        verticalAlign: "top",
        marginLeft: "5px"
      };
      if (x.effect < 0) {
        afterLabel = <span style={afterLabelStyle}>{name}</span>;
        beforeLabelStyle.width =
          40 + this.width - this.scale(Math.abs(x.effect));
        beforeLabelStyle.textAlign = "right";
        beforeLabelStyle.color = "#999";
        beforeLabelStyle.fontSize = "13px";
        beforeLabel = (
          <span style={beforeLabelStyle}>{this.effectFormat(x.effect)}</span>
        );
      } else {
        beforeLabelStyle.textAlign = "right";
        beforeLabel = <span style={beforeLabelStyle}>{name}</span>;
        afterLabelStyle.width = 40;
        afterLabelStyle.textAlign = "left";
        afterLabelStyle.color = "#999";
        afterLabelStyle.fontSize = "13px";
        afterLabel = (
          <span style={afterLabelStyle}>{this.effectFormat(x.effect)}</span>
        );
      }

      return (
        <div key={k} style={{ marginTop: "2px" }}>
          {beforeLabel}
          <div style={style} />
          {afterLabel}
        </div>
      );
    });

    return <span>{rows}</span>;
  }
}

SimpleListVisualizer.defaultProps = {
  plot_cmap: "RdBu"
};

export default SimpleListVisualizer;
