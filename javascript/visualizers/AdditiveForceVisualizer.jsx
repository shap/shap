import React from "react";
import { select } from "d3-selection";
import { scaleLinear } from "d3-scale";
import { format } from "d3-format";
import { axisBottom } from "d3-axis";
import { line } from "d3-shape";
import { hsl } from "d3-color";
import {
  sortBy,
  map,
  each,
  sum,
  filter,
  findIndex,
  debounce
} from "lodash";
import colors from "../color-set";

class AdditiveForceVisualizer extends React.Component {
  constructor() {
    super();
    window.lastAdditiveForceVisualizer = this;
    this.effectFormat = format(".2");
    this.redraw = debounce(() => this.draw(), 200);
  }

  componentDidMount() {
    // create our permanent elements
    this.mainGroup = this.svg.append("g");
    this.axisElement = this.mainGroup
      .append("g")
      .attr("transform", "translate(0,35)")
      .attr("class", "force-bar-axis");
    this.onTopGroup = this.svg.append("g");
    this.baseValueTitle = this.svg.append("text");
    this.joinPointLine = this.svg.append("line");
    this.joinPointLabelOutline = this.svg.append("text");
    this.joinPointLabel = this.svg.append("text");
    this.joinPointTitleLeft = this.svg.append("text");
    this.joinPointTitleLeftArrow = this.svg.append("text");
    this.joinPointTitle = this.svg.append("text");
    this.joinPointTitleRightArrow = this.svg.append("text");
    this.joinPointTitleRight = this.svg.append("text");

    // Define the tooltip objects
    this.hoverLabelBacking = this.svg.append("text")
      .attr("x", 10)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .attr("font-size", 12)
      .attr("stroke", "#fff")
      .attr("fill", "#fff")
      .attr("stroke-width", "4")
      .attr("stroke-linejoin", "round")
      .text("")
      .on("mouseover", () => {
        this.hoverLabel.attr("opacity", 1);
        this.hoverLabelBacking.attr("opacity", 1);
      })
      .on("mouseout", () => {
        this.hoverLabel.attr("opacity", 0);
        this.hoverLabelBacking.attr("opacity", 0);
      });
    this.hoverLabel = this.svg.append("text")
      .attr("x", 10)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .attr("font-size", 12)
      .attr("fill", "#0f0")
      .text("")
      .on("mouseover", () => {
        this.hoverLabel.attr("opacity", 1);
        this.hoverLabelBacking.attr("opacity", 1);
      })
      .on("mouseout", () => {
        this.hoverLabel.attr("opacity", 0);
        this.hoverLabelBacking.attr("opacity", 0);
      });

    // Create our colors and color gradients
    // Verify custom color map
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
    this.colors = plot_colors.map(x => hsl(x));
    this.brighterColors = [1.45, 1.6].map((v, i) => this.colors[i].brighter(v));
    this.colors.map((c, i) => {
      let grad = this.svg
        .append("linearGradient")
        .attr("id", "linear-grad-" + i)
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "0%")
        .attr("y2", "100%");
      grad
        .append("stop")
        .attr("offset", "0%")
        .attr("stop-color", c)
        .attr("stop-opacity", 0.6);
      grad
        .append("stop")
        .attr("offset", "100%")
        .attr("stop-color", c)
        .attr("stop-opacity", 0);
      let grad2 = this.svg
        .append("linearGradient")
        .attr("id", "linear-backgrad-" + i)
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "0%")
        .attr("y2", "100%");
      grad2
        .append("stop")
        .attr("offset", "0%")
        .attr("stop-color", c)
        .attr("stop-opacity", 0.5);
      grad2
        .append("stop")
        .attr("offset", "100%")
        .attr("stop-color", c)
        .attr("stop-opacity", 0);
    });

    // create our x axis
    this.tickFormat = format(",.4");
    this.scaleCentered = scaleLinear();
    this.axis = axisBottom()
      .scale(this.scaleCentered)
      .tickSizeInner(4)
      .tickSizeOuter(0)
      .tickFormat(d => this.tickFormat(this.invLinkFunction(d)))
      .tickPadding(-18);

    // draw and then listen for resize events
    //this.draw();
    window.addEventListener("resize", this.redraw);
    window.setTimeout(this.redraw, 50); // re-draw after interface has updated
  }

  componentDidUpdate() {
    this.draw();
  }

  draw() {
    // copy the feature names onto the features
    each(this.props.featureNames, (n, i) => {
      if (this.props.features[i]) this.props.features[i].name = n;
    });

    // create our link function
    if (this.props.link === "identity") {
      this.invLinkFunction = x => this.props.baseValue + x;
    } else if (this.props.link === "logit") {
      this.invLinkFunction = x =>
        1 / (1 + Math.exp(-(this.props.baseValue + x))); // logistic is inverse of logit
    } else {
      console.log("ERROR: Unrecognized link function: ", this.props.link);
    }

    // Set the dimensions of the plot
    let width = this.svg.node().parentNode.offsetWidth;
    if (width == 0) return setTimeout(() => this.draw(this.props), 500);
    this.svg.style("height", 150 + "px");
    this.svg.style("width", width + "px");
    let topOffset = 50;

    let data = sortBy(this.props.features, x => -1 / (x.effect + 1e-10));
    let totalEffect = sum(map(data, x => Math.abs(x.effect)));
    let totalPosEffects =
      sum(map(filter(data, x => x.effect > 0), x => x.effect)) || 0;
    let totalNegEffects =
      sum(map(filter(data, x => x.effect < 0), x => -x.effect)) || 0;
    this.domainSize = Math.max(totalPosEffects, totalNegEffects) * 3;
    let scale = scaleLinear()
      .domain([0, this.domainSize])
      .range([0, width]);
    let scaleOffset = width / 2 - scale(totalNegEffects);

    this.scaleCentered
      .domain([-this.domainSize / 2, this.domainSize / 2])
      .range([0, width])
      .clamp(true);
    this.axisElement
      .attr("transform", "translate(0," + topOffset + ")")
      .call(this.axis);

    // calculate the position of the join point between positive and negative effects
    // and also the positions of each feature effect block
    let pos = 0,
      i,
      joinPoint,
      joinPointIndex;
    for (i = 0; i < data.length; ++i) {
      data[i].x = pos;
      if (data[i].effect < 0 && joinPoint === undefined) {
        joinPoint = pos;
        joinPointIndex = i;
      }
      pos += Math.abs(data[i].effect);
    }
    if (joinPoint === undefined) {
      joinPoint = pos;
      joinPointIndex = i;
    }

    let lineFunction = line()
      .x(d => d[0])
      .y(d => d[1]);

    let getLabel = d => {
      if (d.value !== undefined && d.value !== null && d.value !== "") {
        return (
          d.name +
          " = " +
          (isNaN(d.value) ? d.value : this.tickFormat(d.value))
        );
      } else return d.name;
    };

    data = this.props.hideBars ? [] : data;
    let blocks = this.mainGroup.selectAll(".force-bar-blocks").data(data);
    blocks
      .enter()
      .append("path")
      .attr("class", "force-bar-blocks")
      .merge(blocks)
      .attr("d", (d, i) => {
        let x = scale(d.x) + scaleOffset;
        let w = scale(Math.abs(d.effect));
        let pointShiftStart = d.effect < 0 ? -4 : 4;
        let pointShiftEnd = pointShiftStart;
        if (i === joinPointIndex) pointShiftStart = 0;
        if (i === joinPointIndex - 1) pointShiftEnd = 0;
        return lineFunction([
          [x, 6 + topOffset],
          [x + w, 6 + topOffset],
          [x + w + pointShiftEnd, 14.5 + topOffset],
          [x + w, 23 + topOffset],
          [x, 23 + topOffset],
          [x + pointShiftStart, 14.5 + topOffset]
        ]);
      })
      .attr("fill", d => (d.effect > 0 ? this.colors[0] : this.colors[1]))
      .on("mouseover", d => {
        if (scale(Math.abs(d.effect)) < scale(totalEffect) / 50 ||
            scale(Math.abs(d.effect)) < 10) {
          let x = scale(d.x) + scaleOffset;
          let w = scale(Math.abs(d.effect));
          this.hoverLabel
            .attr("opacity", 1)
            .attr("x", x + w/2)
            .attr("y", topOffset + 0.5)
            .attr("fill", d.effect > 0 ? this.colors[0] : this.colors[1])
            .text(getLabel(d));
          this.hoverLabelBacking
            .attr("opacity", 1)
            .attr("x", x + w/2)
            .attr("y", topOffset + 0.5)
            .text(getLabel(d));
        }
      })
      .on("mouseout", () => {
        this.hoverLabel.attr("opacity", 0);
        this.hoverLabelBacking.attr("opacity", 0);
      });
    blocks.exit().remove();

    let filteredData = _.filter(data, d => {
      return (
        scale(Math.abs(d.effect)) > scale(totalEffect) / 50 &&
        scale(Math.abs(d.effect)) > 10
      );
    });

    let labels = this.onTopGroup
      .selectAll(".force-bar-labels")
      .data(filteredData);
    labels.exit().remove();
    labels = labels
      .enter()
      .append("text")
      .attr("class", "force-bar-labels")
      .attr("font-size", "12px")
      .attr("y", 48 + topOffset)
      .merge(labels)
      .text(d => {
        if (d.value !== undefined && d.value !== null && d.value !== "") {
          return (
            d.name +
            " = " +
            (isNaN(d.value) ? d.value : this.tickFormat(d.value))
          );
        } else return d.name;
      })
      .attr("fill", d => (d.effect > 0 ? this.colors[0] : this.colors[1]))
      .attr("stroke", function(d) {
        d.textWidth = Math.max(
          this.getComputedTextLength(),
          scale(Math.abs(d.effect)) - 10
        );
        d.innerTextWidth = this.getComputedTextLength();
        return "none";
      });
    this.filteredData = filteredData;

    // compute where the text labels should go
    if (data.length > 0) {
      pos = joinPoint + scale.invert(5);
      for (let i = joinPointIndex; i < data.length; ++i) {
        data[i].textx = pos;
        pos += scale.invert(data[i].textWidth + 10);
      }
      pos = joinPoint - scale.invert(5);
      for (let i = joinPointIndex - 1; i >= 0; --i) {
        data[i].textx = pos;
        pos -= scale.invert(data[i].textWidth + 10);
      }
    }

    labels
      .attr(
        "x",
        d =>
          scale(d.textx) +
          scaleOffset +
          (d.effect > 0 ? -d.textWidth / 2 : d.textWidth / 2)
      )
      .attr("text-anchor", "middle"); //d => d.effect > 0 ? 'end' : 'start');

    // Now that we know the text widths we further filter by what fits on the screen
    filteredData = filter(filteredData, d => {
      return (
        scale(d.textx) + scaleOffset > this.props.labelMargin &&
        scale(d.textx) + scaleOffset < width - this.props.labelMargin
      );
    });
    this.filteredData2 = filteredData;

    // Build an array with one extra feature added
    let filteredDataPlusOne = filteredData.slice();
    let ind = findIndex(data, filteredData[0]) - 1;
    if (ind >= 0) filteredDataPlusOne.unshift(data[ind]);

    let labelBacking = this.mainGroup
      .selectAll(".force-bar-labelBacking")
      .data(filteredData);
    labelBacking
      .enter()
      .append("path")
      .attr("class", "force-bar-labelBacking")
      .attr("stroke", "none")
      .attr("opacity", 0.2)
      .merge(labelBacking)
      .attr("d", d => {
        return lineFunction([
          [
            scale(d.x) + scale(Math.abs(d.effect)) + scaleOffset,
            23 + topOffset
          ],
          [
            (d.effect > 0 ? scale(d.textx) : scale(d.textx) + d.textWidth) +
              scaleOffset +
              5,
            33 + topOffset
          ],
          [
            (d.effect > 0 ? scale(d.textx) : scale(d.textx) + d.textWidth) +
              scaleOffset +
              5,
            54 + topOffset
          ],
          [
            (d.effect > 0 ? scale(d.textx) - d.textWidth : scale(d.textx)) +
              scaleOffset -
              5,
            54 + topOffset
          ],
          [
            (d.effect > 0 ? scale(d.textx) - d.textWidth : scale(d.textx)) +
              scaleOffset -
              5,
            33 + topOffset
          ],
          [scale(d.x) + scaleOffset, 23 + topOffset]
        ]);
      })
      .attr("fill", d => `url(#linear-backgrad-${d.effect > 0 ? 0 : 1})`);
    labelBacking.exit().remove();

    let labelDividers = this.mainGroup
      .selectAll(".force-bar-labelDividers")
      .data(filteredData.slice(0, -1));
    labelDividers
      .enter()
      .append("rect")
      .attr("class", "force-bar-labelDividers")
      .attr("height", "21px")
      .attr("width", "1px")
      .attr("y", 33 + topOffset)
      .merge(labelDividers)
      .attr(
        "x",
        d =>
          (d.effect > 0 ? scale(d.textx) : scale(d.textx) + d.textWidth) +
          scaleOffset +
          4.5
      )
      .attr("fill", d => `url(#linear-grad-${d.effect > 0 ? 0 : 1})`);
    labelDividers.exit().remove();

    let labelLinks = this.mainGroup
      .selectAll(".force-bar-labelLinks")
      .data(filteredData.slice(0, -1));
    labelLinks
      .enter()
      .append("line")
      .attr("class", "force-bar-labelLinks")
      .attr("y1", 23 + topOffset)
      .attr("y2", 33 + topOffset)
      .attr("stroke-opacity", 0.5)
      .attr("stroke-width", 1)
      .merge(labelLinks)
      .attr("x1", d => scale(d.x) + scale(Math.abs(d.effect)) + scaleOffset)
      .attr(
        "x2",
        d =>
          (d.effect > 0 ? scale(d.textx) : scale(d.textx) + d.textWidth) +
          scaleOffset +
          5
      )
      .attr("stroke", d => (d.effect > 0 ? this.colors[0] : this.colors[1]));
    labelLinks.exit().remove();

    let blockDividers = this.mainGroup
      .selectAll(".force-bar-blockDividers")
      .data(data.slice(0, -1));
    blockDividers
      .enter()
      .append("path")
      .attr("class", "force-bar-blockDividers")
      .attr("stroke-width", 2)
      .attr("fill", "none")
      .merge(blockDividers)
      .attr("d", d => {
        let pos = scale(d.x) + scale(Math.abs(d.effect)) + scaleOffset;
        return lineFunction([
          [pos, 6 + topOffset],
          [pos + (d.effect < 0 ? -4 : 4), 14.5 + topOffset],
          [pos, 23 + topOffset]
        ]);
      })
      .attr("stroke", (d, i) => {
        if (joinPointIndex === i + 1 || Math.abs(d.effect) < 1e-8)
          return "#rgba(0,0,0,0)";
        else if (d.effect > 0) return this.brighterColors[0];
        else return this.brighterColors[1];
      });
    blockDividers.exit().remove();

    this.joinPointLine
      .attr("x1", scale(joinPoint) + scaleOffset)
      .attr("x2", scale(joinPoint) + scaleOffset)
      .attr("y1", 0 + topOffset)
      .attr("y2", 6 + topOffset)
      .attr("stroke", "#F2F2F2")
      .attr("stroke-width", 1)
      .attr("opacity", 1);

    this.joinPointLabelOutline
      .attr("x", scale(joinPoint) + scaleOffset)
      .attr("y", -5 + topOffset)
      .attr("color", "#fff")
      .attr("text-anchor", "middle")
      .attr("font-weight", "bold")
      .attr("stroke", "#fff")
      .attr("stroke-width", 6)
      .text(format(",.2f")(this.invLinkFunction(joinPoint - totalNegEffects)))
      .attr("opacity", 1);
    console.log(
      "joinPoint",
      joinPoint,
      scaleOffset,
      topOffset,
      totalNegEffects
    );
    this.joinPointLabel
      .attr("x", scale(joinPoint) + scaleOffset)
      .attr("y", -5 + topOffset)
      .attr("text-anchor", "middle")
      .attr("font-weight", "bold")
      .attr("fill", "#000")
      .text(format(",.2f")(this.invLinkFunction(joinPoint - totalNegEffects)))
      .attr("opacity", 1);

    this.joinPointTitle
      .attr("x", scale(joinPoint) + scaleOffset)
      .attr("y", -22 + topOffset)
      .attr("text-anchor", "middle")
      .attr("font-size", "12")
      .attr("fill", "#000")
      .text(this.props.outNames[0])
      .attr("opacity", 0.5);

    if (!this.props.hideBars) {
      this.joinPointTitleLeft
        .attr("x", scale(joinPoint) + scaleOffset - 16)
        .attr("y", -38 + topOffset)
        .attr("text-anchor", "end")
        .attr("font-size", "13")
        .attr("fill", this.colors[0])
        .text("higher")
        .attr("opacity", 1.0);
      this.joinPointTitleRight
        .attr("x", scale(joinPoint) + scaleOffset + 16)
        .attr("y", -38 + topOffset)
        .attr("text-anchor", "start")
        .attr("font-size", "13")
        .attr("fill", this.colors[1])
        .text("lower")
        .attr("opacity", 1.0);

      this.joinPointTitleLeftArrow
        .attr("x", scale(joinPoint) + scaleOffset + 7)
        .attr("y", -42 + topOffset)
        .attr("text-anchor", "end")
        .attr("font-size", "13")
        .attr("fill", this.colors[0])
        .text("→")
        .attr("opacity", 1.0);
      this.joinPointTitleRightArrow
        .attr("x", scale(joinPoint) + scaleOffset - 7)
        .attr("y", -36 + topOffset)
        .attr("text-anchor", "start")
        .attr("font-size", "13")
        .attr("fill", this.colors[1])
        .text("←")
        .attr("opacity", 1.0);
    }
    if (!this.props.hideBaseValueLabel) {
      this.baseValueTitle
        .attr("x", this.scaleCentered(0))
        .attr("y", -22 + topOffset)
        .attr("text-anchor", "middle")
        .attr("font-size", "12")
        .attr("fill", "#000")
        .text("base value")
        .attr("opacity", 0.5);
    }
  }

  componentWillUnmount() {
    window.removeEventListener("resize", this.redraw);
  }

  render() {
    return (
      <svg
        ref={x => (this.svg = select(x))}
        style={{
          userSelect: "none",
          display: "block",
          fontFamily: "arial",
          sansSerif: true
        }}
      >
        <style
          dangerouslySetInnerHTML={{
            __html: `
          .force-bar-axis path {
            fill: none;
            opacity: 0.4;
          }
          .force-bar-axis paths {
            display: none;
          }
          .tick line {
            stroke: #000;
            stroke-width: 1px;
            opacity: 0.4;
          }
          .tick text {
            fill: #000;
            opacity: 0.5;
            font-size: 12px;
            padding: 0px;
          }`
          }}
        />
      </svg>
    );
  }
}

AdditiveForceVisualizer.defaultProps = {
  plot_cmap: "RdBu"
};

export default AdditiveForceVisualizer;
