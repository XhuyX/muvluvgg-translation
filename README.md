# muvluvgg-translation
Simple Vietnamese translation for Muvluv Girls Garden
# Unity TMP Font Fix for Vietnamese (Sharp Outline)

Fix lỗi hiển thị tiếng Việt bị mờ và răng cưa khi bật Outline trong game Unity sử dụng TextMeshPro.
Fixes blurred Vietnamese characters and jagged outlines in Unity games using TextMeshPro.

## Vấn đề / The Problem
- Default font config causes merging characters when Outline is enabled.
- Low sampling size makes text look blurry.

## Giải pháp / Solution Config
- **Font:** Consolas (CONSOLA SDF)
- **Sampling Point Size:** 64 (Custom)
- **Padding:** 7
- **Atlas Resolution:** 2048x2048
- **Render Mode:** SDFAA
