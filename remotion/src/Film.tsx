import React, {useEffect, useState} from 'react';
import {
  AbsoluteFill,
  Audio,
  Easing,
  OffthreadVideo,
  Sequence,
  continueRender,
  delayRender,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {useAudioData, visualizeAudio} from '@remotion/media-utils';
import type {FilmProps, Line, Shot} from './types';

// ------------------------------------------------------------------ fonts
const FONTS: [string, string, string, string][] = [
  ['Inter', 'Inter-Regular.otf', '400', 'normal'],
  ['Inter', 'Inter-Medium.otf', '500', 'normal'],
  ['Inter', 'Inter-SemiBold.otf', '600', 'normal'],
  ['Instrument Serif', 'instrument-serif-latin-400-normal.woff2', '400', 'normal'],
  ['Instrument Serif', 'instrument-serif-latin-400-italic.woff2', '400', 'italic'],
];

const useFonts = () => {
  const [handle] = useState(() => delayRender('fonts'));
  useEffect(() => {
    Promise.all(
      FONTS.map(([family, file, weight, style]) => {
        const face = new FontFace(family, `url(${staticFile(`fonts/${file}`)})`, {weight, style});
        (document.fonts as unknown as {add: (f: FontFace) => void}).add(face);
        return face.load();
      }),
    )
      .then(() => continueRender(handle))
      .catch((e) => {
        console.error(e);
        continueRender(handle);
      });
  }, [handle]);
};

const SANS = 'Inter, sans-serif';
const SERIF = '"Instrument Serif", Georgia, serif';
const PAPER = '#F4EFE6';

// ------------------------------------------------------------------ footage
const ShotView: React.FC<{shot: Shot}> = ({shot}) => {
  const frame = useCurrentFrame();
  // a slow, barely-there push keeps still moments alive, like a camera operator breathing
  const scale = interpolate(frame, [0, shot.duration], [1, 1 + (shot.push ?? 0.025)], {
    extrapolateRight: 'clamp',
  });
  return (
    <AbsoluteFill style={{overflow: 'hidden', backgroundColor: 'black'}}>
      <OffthreadVideo
        src={staticFile(shot.src)}
        muted
        style={{width: '100%', height: '100%', objectFit: 'cover', transform: `scale(${scale})`}}
      />
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------ voice message UI
type PillSegment = {from: number; duration: number; label: string; fx: string; items: Line[]};

const pillSegments = (lines: Line[]): PillSegment[] => {
  const segs: PillSegment[] = [];
  for (const l of lines) {
    if (!['voicenote', 'phone', 'radio'].includes(l.fx)) continue;
    const last = segs[segs.length - 1];
    if (last && last.label === l.label && last.fx === l.fx && l.from - (last.from + last.duration) < 40) {
      last.duration = l.from + l.duration - last.from;
      last.items.push(l);
    } else {
      segs.push({from: l.from, duration: l.duration, label: l.label, fx: l.fx, items: [l]});
    }
  }
  return segs;
};

const BARS = 30;

const LiveBars: React.FC<{src: string}> = ({src}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const audio = useAudioData(staticFile(src));
  if (!audio) return <FlatBars />;
  const values = visualizeAudio({fps, frame, audioData: audio, numberOfSamples: 64, smoothing: true});
  // speech energy sits low in the spectrum: spread the lower half over the bars, mirrored
  const half = values.slice(1, BARS / 2 + 1);
  const bars = [...[...half].reverse(), ...half];
  return <Bars heights={bars.map((v) => Math.min(1, Math.sqrt(v) * 2.4))} />;
};

const FlatBars: React.FC = () => <Bars heights={new Array(BARS).fill(0.06)} />;

const Bars: React.FC<{heights: number[]}> = ({heights}) => (
  <div style={{display: 'flex', alignItems: 'center', gap: 5, height: 56}}>
    {heights.map((h, i) => (
      <div
        key={i}
        style={{
          width: 6,
          height: Math.max(5, h * 56),
          borderRadius: 3,
          backgroundColor: PAPER,
          opacity: 0.92,
        }}
      />
    ))}
  </div>
);

const pillStyle: Record<string, {dot: string; caption: string}> = {
  voicenote: {dot: '#8FD3A4', caption: 'Voice message'},
  phone: {dot: '#8FD3A4', caption: 'On call'},
  radio: {dot: '#FF5A4E', caption: 'On air'},
};

const VoicePill: React.FC<{seg: PillSegment}> = ({seg}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({frame, fps, config: {damping: 200, mass: 0.6}});
  const exit = interpolate(frame, [seg.duration - 2, seg.duration + 6], [1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const secs = Math.max(0, Math.floor(frame / fps));
  const st = pillStyle[seg.fx] ?? pillStyle.voicenote;
  return (
    <AbsoluteFill style={{alignItems: 'center', top: 168}}>
      <div
        style={{
          opacity: enter * exit,
          transform: `translateY(${(1 - enter) * -24}px) scale(${0.96 + 0.04 * enter})`,
          display: 'flex',
          alignItems: 'center',
          gap: 22,
          padding: '20px 30px 20px 22px',
          borderRadius: 999,
          backgroundColor: 'rgba(16,16,18,0.58)',
          backdropFilter: 'blur(22px)',
          border: '1px solid rgba(255,255,255,0.10)',
          boxShadow: '0 18px 50px rgba(0,0,0,0.35)',
          color: PAPER,
          fontFamily: SANS,
          height: 'fit-content',
        }}
      >
        <div
          style={{
            width: 70,
            height: 70,
            borderRadius: 999,
            background: 'linear-gradient(140deg, #D9C3A5, #8C6E57)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 32,
            fontWeight: 600,
            color: '#1d1712',
          }}
        >
          {seg.label.slice(0, 1).toUpperCase()}
        </div>
        <div style={{display: 'flex', flexDirection: 'column', gap: 6}}>
          <div style={{display: 'flex', alignItems: 'center', gap: 10, fontSize: 25, fontWeight: 600}}>
            <span
              style={{
                width: 11,
                height: 11,
                borderRadius: 99,
                backgroundColor: st.dot,
                opacity: 0.55 + 0.45 * Math.abs(Math.sin(frame / 7)),
              }}
            />
            {seg.label}
            <span style={{fontWeight: 400, opacity: 0.6, marginLeft: 4}}>· {st.caption}</span>
          </div>
          <div style={{position: 'relative', height: 56}}>
            {seg.items.some((l) => frame >= l.from - seg.from && frame < l.from - seg.from + l.duration) ? null : (
              <FlatBars />
            )}
            {seg.items.map((l, i) => (
              <Sequence key={i} from={l.from - seg.from} durationInFrames={l.duration} layout="none">
                <div style={{position: 'absolute', inset: 0}}>
                  <LiveBars src={l.src} />
                </div>
              </Sequence>
            ))}
          </div>
        </div>
        <div style={{fontSize: 24, fontVariantNumeric: 'tabular-nums', opacity: 0.7, marginLeft: 6}}>
          0:{String(secs).padStart(2, '0')}
        </div>
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------ subtitles
const Subtitle: React.FC<{line: Line}> = ({line}) => {
  const frame = useCurrentFrame();
  const o = interpolate(frame, [0, 4, line.duration - 3, line.duration + 2], [0, 1, 1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <AbsoluteFill style={{justifyContent: 'flex-start', alignItems: 'center', top: 1236}}>
      <div
        style={{
          opacity: o,
          maxWidth: 880,
          textAlign: 'center',
          fontFamily: SANS,
          fontWeight: 500,
          fontSize: 46,
          lineHeight: 1.3,
          color: PAPER,
          letterSpacing: '-0.005em',
          textShadow: '0 2px 3px rgba(0,0,0,0.75), 0 0 22px rgba(0,0,0,0.55)',
          padding: '0 40px',
        }}
      >
        {line.text}
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------ title
const Title: React.FC<{p: FilmProps}> = ({p}) => {
  const frame = useCurrentFrame();
  const d = p.titleDuration;
  const o = interpolate(frame, [0, 10, d - 10, d], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const rise = interpolate(frame, [0, 16], [14, 0], {extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic)});
  const track = interpolate(frame, [0, d], [0.36, 0.46]);
  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', top: -120, opacity: o}}>
      <AbsoluteFill
        style={{background: 'radial-gradient(ellipse 70% 22% at 50% 50%, rgba(0,0,0,0.42), rgba(0,0,0,0))'}}
      />
      <div style={{transform: `translateY(${rise}px)`, textAlign: 'center', color: PAPER}}>
        <div style={{fontFamily: SANS, fontWeight: 600, fontSize: 24, letterSpacing: `${track}em`, opacity: 0.85}}>
          {p.series} · Nº {String(p.episode).padStart(2, '0')}
        </div>
        <div
          style={{
            fontFamily: SERIF,
            fontStyle: 'italic',
            fontSize: 104,
            lineHeight: 1.05,
            marginTop: 18,
            textShadow: '0 4px 30px rgba(0,0,0,0.45)',
          }}
        >
          {p.title}
        </div>
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------ end card
const EndCard: React.FC<{p: FilmProps; duration: number}> = ({p, duration}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const bg = interpolate(frame, [0, 8], [0, 1], {extrapolateRight: 'clamp'});
  const item = (delay: number) => {
    const s = spring({frame: frame - delay, fps, config: {damping: 200}});
    return {opacity: s, transform: `translateY(${(1 - s) * 16}px)`};
  };
  const out = interpolate(frame, [duration - 6, duration], [1, 0.0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return (
    <AbsoluteFill style={{backgroundColor: `rgba(14,13,12,${bg})`, alignItems: 'center', justifyContent: 'center'}}>
      <div style={{textAlign: 'center', color: PAPER, opacity: out}}>
        <div style={{...item(4), fontFamily: SANS, fontWeight: 600, fontSize: 26, letterSpacing: '0.5em'}}>
          {p.series}
        </div>
        <div style={{...item(8), width: 120, height: 1, backgroundColor: PAPER, opacity: 0.35, margin: '34px auto'}} />
        <div style={{...item(12), fontFamily: SERIF, fontStyle: 'italic', fontSize: 84, lineHeight: 1.05}}>
          {p.tagline}
        </div>
        <div style={{...item(20), fontFamily: SANS, fontSize: 30, marginTop: 46, opacity: 0.8}}>{p.handle}</div>
        <div style={{...item(26), fontFamily: SANS, fontSize: 20, marginTop: 16, opacity: 0.45, letterSpacing: '0.2em'}}>
          MADE WITH AI
        </div>
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------ film
export const Film: React.FC<FilmProps> = (p) => {
  useFonts();
  const segs = pillSegments(p.lines);
  return (
    <AbsoluteFill style={{backgroundColor: 'black'}}>
      {p.shots.map((s, i) => (
        <Sequence key={`shot${i}`} from={s.from} durationInFrames={s.duration}>
          <ShotView shot={s} />
        </Sequence>
      ))}
      <Sequence from={p.titleFrom} durationInFrames={p.titleDuration}>
        <Title p={p} />
      </Sequence>
      {segs.map((seg, i) => (
        <Sequence key={`pill${i}`} from={seg.from - 3} durationInFrames={seg.duration + 10}>
          <VoicePill seg={{...seg}} />
        </Sequence>
      ))}
      {p.lines.map((l, i) => (
        <Sequence key={`sub${i}`} from={l.from} durationInFrames={l.duration + 3}>
          <Subtitle line={l} />
        </Sequence>
      ))}
      <Sequence from={p.storyFrames} durationInFrames={p.durationInFrames - p.storyFrames}>
        <EndCard p={p} duration={p.durationInFrames - p.storyFrames} />
      </Sequence>
      {p.audio ? <Audio src={staticFile(p.audio)} /> : null}
    </AbsoluteFill>
  );
};
