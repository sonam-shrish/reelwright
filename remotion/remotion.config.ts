import {Config} from '@remotion/cli/config';

// The sandbox has Playwright's headless Chromium; use it instead of downloading one.
Config.setBrowserExecutable(
  process.env.REMOTION_CHROME ??
    '/opt/pw-browsers/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell',
);
Config.setVideoImageFormat('jpeg');
Config.setJpegQuality(95);
Config.setCodec('h264');
Config.setCrf(16);
Config.setPixelFormat('yuv420p');
Config.setConcurrency(4);
