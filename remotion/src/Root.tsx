import {Composition, staticFile} from 'remotion';
import {Film} from './Film';
import type {FilmProps} from './types';

const sample: FilmProps = {
  series: 'VOICE NOTES',
  episode: 1,
  title: 'Is it recording?',
  tagline: 'Say it while you can.',
  handle: '@_babelground',
  fps: 24,
  durationInFrames: 720,
  storyFrames: 660,
  audio: '',
  shots: [],
  lines: [],
  titleFrom: 14,
  titleDuration: 60,
};

export const RemotionRoot: React.FC = () => (
  <Composition
    id="VoiceNote"
    component={Film}
    width={1080}
    height={1920}
    fps={24}
    durationInFrames={720}
    defaultProps={sample}
    calculateMetadata={({props}) => ({durationInFrames: props.durationInFrames, fps: props.fps})}
  />
);

export const fontUrl = (f: string) => staticFile(`fonts/${f}`);
