import React, { Component } from 'react';
import ImageLoader from './imageLoader';
import Chart from 'chart.js';
import { calculateHistogram } from './util';

class App extends Component {
  constructor(props) {
    super(props);
 
    this.state = {image: undefined};

    this.drawHistogram = this.drawHistogram.bind(this);
    this.imageChange = this.imageChange.bind(this);
  }

  async drawHistogram() {
    if (!this.histogramCanvas) {
      return;
    }
    const canvas = this.histogramCanvas;
    const ctx = canvas.getContext('2d');
    const histograms = await calculateHistogram(this.state.image);

    new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array.apply(null, {length: 256}).map(Number.call, Number),
        datasets: [{
          label: "Red Channel",
          data: histograms.red,
          borderColor: "#ff0000",
        }, {
          label: "Green Channel",
          data: histograms.green,
          borderColor: "#00ff00",
        }, {
          label: "Blue Channel",
          data: histograms.blue,
          borderColor: "#0000ff",
        }, {
          label: "Grayscale",
          data: histograms.grayscale,
          borderColor: "#000000",
        }]
      },
      options: {
        scales: {
          xAxes: [{
            display: true,
            ticks: {
              callback: (dataLabel, index) => index % 5 === 0 ? dataLabel : ''
            }
          }],
          yAxes: [{display: true, beginAtZero: true}]
        }
      }
    });
  }

  imageChange(event) {
    this.setState({image: event.image});
    this.drawHistogram();
  }

  render() {
    return (
      <div>
        <div style={{marginTop: 10}}>
          <ImageLoader onChange={this.imageChange} />
        </div>
        <div style={{marginTop: 10}}>
          <img ref={v => this.image = v} src={this.state.image} width="100%" />
        </div>
        <div style={{marginTop: 10}}>
          <canvas ref={v => this.histogramCanvas = v} width={400} height={400} />
        </div>
      </div>
    );
  }
}

export default App;
