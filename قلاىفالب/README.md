# Rail Operator for Android

This project keeps the original Pygame train simulator and packages it with
python-for-android/SDL2. The Android build uses Pygame and NumPy; it does not
require `sounddevice`, SciPy, or `pedalboard`. Audio falls back to the Pygame
mixer on Android.

The game is landscape-oriented and includes on-screen touch controls. Hold
POWER or BRAKE to drive; hold SAND, COOL, or HORN for those controls. PAUSE
toggles the pause screen. Loading and unloading use the large cargo button.

## Build an APK on GitHub

1. Open this project folder in VS Code.
2. In the integrated terminal, initialize and push this folder to a new,
   empty GitHub repository:

   ```powershell
   git init
   git add .
   git commit -m "Prepare train simulator for Android"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/rail-operator.git
   git push -u origin main
   ```

3. On GitHub, open the repository's **Actions** tab and allow workflows if
   GitHub asks.
4. Choose **Build Android APK**, select **Run workflow**, and run it on `main`.
   A push to the repository also starts a build.
5. Open the completed workflow run, download the
   `rail-operator-android-apk` artifact, and extract the APK.
6. Transfer the APK to the phone, allow installation from that source when
   Android prompts, then install it.

The build is configured with Android API 36 (Android 16), landscape orientation,
and the 64-bit ARM ABI. GitHub Actions builds on Linux because Buildozer does
not build Android packages natively on Windows.

## Build locally

Buildozer requires Linux or macOS; on Windows, use WSL2 with Ubuntu:

```bash
python3 -m pip install --user buildozer cython==0.29.34
buildozer -v android debug
```

The first build downloads the Android SDK/NDK and can take a while. The debug
APK is written to `bin/`.
