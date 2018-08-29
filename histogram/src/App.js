import React, { Component } from 'react';
import ImageLoader from './imageLoader';

class App extends Component {
  constructor(props) {
    super(props);
 
    this.state = {image: undefined};
  }

  render() {
    return (
      <div>
        <div>
          <ImageLoader onChange={ev => this.setState({image: ev.image})} />
        </div>
        <div>
          <img src={this.state.image} />
        </div>
      </div>
    );
  }
}

export default App;
