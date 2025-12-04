
# Empathy Piano

The empathy piano is a both virtual and physical instrument meant to allow musicians to have their play sound like their feelings.

The physical piano component is first plugged into the user's laptop. After running the piano and loading the sounds and camera, the user is greeted with a virtual piano interface along with an octave counter and a button to adjust the method of tracking emotion.

The default emotion when opening the piano program is neutral; however, by holding left click on the mouse, the user can have the followings emotions read:

- Anger
- Fear
- Happiness
- Sadness
- Surprise
- Disgust
- Neutrality

Each of the emotions comes equipped with their own unique sound that represents the feeling that emotion gives off:

- Anger is a deep loud orchestra hit.
- Fear is an eerie synth effect.
- Happiness is a steel pan.
- Sadness is a reverb heavy grand piano.
- Surprise is a light string pluck.
- Disgust is a weird xylophone hit.
- Neutrality is a generic piano note.

Emotion changing can be done through the use of the left click button each time. Clicking the hold button at the bottom of the screen will change to live mode, which allows the user to have their piano emotion change dynamically.

The base octave the piano is in is 4. To change the octave, the user can give a thumbs up to give an octave higher or a thumbs down to go an octave lower. The user is limited to octaves 3 through 5 for sound quality purposes.

## Installation

The Empathy Piano requires Python, preferrably 3.9, but any version under 3.11 should work.

Once an applicable version of python is installed, you can download the ZIP file of the Github directory and extract it. After extraction, first open the *installRequirements.bat* file. The project uses an array of different python libaries to achieve its visual, aural, and technical features.

*installRequirements.bat* installs each library from the *requirements.txt* file to the applicable version for the project. An error that may occur is if a specific libary is too up to date, as tensorflow is only compatabile with certain older versions of protobuf.

After installing the requirements, you can run the piano with the *runPiano.bat* file. If you have Python 3.9 alongside other python verisons, you can use the *runPiano39.bat* file to force the use of Python 3.9.
    
## Authors

- Taimoor Babar
- Lucas Ainbinder
- Travis Lin
- Eshaan Dubey
- Bashaier Jama
- Gabrielly Pessanha de Araujo

