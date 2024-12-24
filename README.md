# AirBrush - Virtual Air Painting Application

AirBrush is a computer vision-based application that allows users to paint virtually in the air using hand gestures. Using your computer's webcam, you can draw and create digital artwork simply by moving your finger in the air.

## Features

- Real-time hand tracking
- Draw by pinching your thumb and index finger together
- Multi-stroke drawing capability
- Clear canvas by showing all fingers up
- Natural and intuitive interface

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- Mediapipe
- Webcam

## Installation

1. Clone this repository or download the source code: https://github.com/DhruvGorasiya/AirBrush

2. Once the application starts:
   - Position your hand in front of the webcam
   - Pinch your thumb and index finger together to start drawing
   - Move your hand while maintaining the pinch to draw
   - Release the pinch to stop drawing
   - Show all fingers up to clear the canvas
   - Press 'q' to quit the application

## Controls

- **Draw**: Pinch thumb and index finger together
- **Stop Drawing**: Release the pinch
- **Clear Canvas**: Show all fingers up
- **Exit**: Press 'q' key

## Technical Details

The application uses:
- MediaPipe for hand landmark detection
- OpenCV for image processing and display
- Real-time finger tracking for gesture recognition
- Distance calculation between thumb and index finger for draw trigger

## Limitations

- Works best in good lighting conditions
- Currently supports single-hand drawing
- Requires a webcam with decent resolution

## Contributing

Feel free to fork this project and submit pull requests for any improvements.

## Author

Dhruv Gorasiya

*Note: This project is for educational purposes and can be modified for personal use.*
