# Baseball Savant Dashboard — Mobile App

A native Android/iOS wrapper around the live dashboard, built with
[Capacitor](https://capacitorjs.com/). The app itself has no game logic or
data of its own — it opens a native WebView pointed at the deployed Flask
site (`app.py` in the repo root), so the app and the web dashboard always
show the same data.

## How it's configured

- `capacitor.config.json` sets `server.url` to the deployed dashboard
  (`https://baseball-savant-dashboard.onrender.com`). Update this if the
  site moves to a different host.
- `assets/` holds the source icon (`icon.png`, `icon-foreground.png`,
  `icon-background.png`) and splash screen (`splash.png`) used to generate
  every platform-specific size via `@capacitor/assets`.
- `android/` and `ios/` are the generated native projects. Don't hand-edit
  generated resource files (`res/mipmap-*`, `Assets.xcassets`) — regenerate
  them instead (see below).
- `www/` is a minimal fallback page bundled into the app; it's not what
  users see in normal operation since `server.url` takes over on launch.

## Building

```bash
cd mobile
npm install
```

### Android

Requires Android Studio (or the Android SDK + a JDK) installed locally —
this project was scaffolded in an environment without the SDK, so it has
not been compiled here.

```bash
npx cap open android
```

Android Studio will sync Gradle and let you Run on a device/emulator, or
Build > Generate Signed Bundle/APK for a release build to upload to the
Play Store.

### iOS

Requires a Mac with Xcode and CocoaPods installed.

```bash
npx cap open ios
```

Xcode will open the workspace; set your signing team under
Signing & Capabilities, then Run on a device/simulator or
Product > Archive for App Store submission.

## Updating the app icon / splash screen

Edit the source images in `assets/`, then regenerate every platform size:

```bash
npx capacitor-assets generate --android --ios
npx cap sync
```

## Pointing at a different backend

If the dashboard is redeployed to a new URL, update `server.url` in
`capacitor.config.json` and re-run `npx cap sync`.
