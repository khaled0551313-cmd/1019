[app]
title = Rail Operator
package.name = railoperator
package.domain = org.khaled
source.dir = .
source.include_exts = py
version = 1.0.0
requirements = python3,pygame,numpy
orientation = landscape
fullscreen = 1

android.api = 36
android.minapi = 23
android.ndk = 28c
android.archs = arm64-v8a
android.accept_sdk_license = True

p4a.bootstrap = sdl2
p4a.branch = develop

[buildozer]
log_level = 2
warn_on_root = 1
