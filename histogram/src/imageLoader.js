import React, { Component } from 'react';
import PropTypes from 'prop-types';

class ImageLoader extends Component {

  static propTypes = {
    onChange: PropTypes.func
  };
  
  constructor(props) {
    super(props);

    this.state = {image: undefined};

    this.imageChange = this.imageChange.bind(this);
  }

  imageChange(ev) {
    if (ev.target.files && ev.target.files[0]) {
      const reader = new FileReader();
      reader.onload = e => {
        this.setState({image: e.target.result});
        if (this.props.onChange) {
          this.props.onChange({
            image: this.state.image,
          });
        }
      }
      reader.readAsDataURL(ev.target.files[0]);
    }
  }

  render() {
    return (
      <input type="file" accept="image/*" capture="camera" onChange={this.imageChange} />
    );
  }

}

export default ImageLoader;
