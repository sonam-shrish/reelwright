export type Shot = {src: string; from: number; duration: number; push?: number};
export type Line = {
  from: number;
  duration: number;
  text: string;
  who: string;
  label: string;
  fx: string; // "voicenote" | "phone" | "radio" | "room" | "" ...
  src: string; // this line's own audio, for the waveform
};
export type FilmProps = {
  series: string;
  episode: number;
  title: string;
  tagline: string;
  handle: string;
  fps: number;
  durationInFrames: number;
  storyFrames: number;
  audio: string;
  shots: Shot[];
  lines: Line[];
  titleFrom: number;
  titleDuration: number;
};
