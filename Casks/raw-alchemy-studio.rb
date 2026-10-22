cask "raw-alchemy-studio" do
  # Update both fields from the final macOS release and its SHA256SUMS.txt.
  version "0.6.4"
  sha256 "35548afdd5bdfdfb8645aa1ca58df89faad93cff47e0ddcd16fee15517d962b0"

  url "https://github.com/shenmintao/Raw-Alchemy/releases/download/v#{version}/RawAlchemyStudio-v#{version}-macos.dmg"
  name "Raw Alchemy Studio"
  desc "Local RAW photo culling and LUT grading"
  homepage "https://github.com/shenmintao/Raw-Alchemy"

  livecheck do
    url :homepage
    strategy :github_latest
  end

  depends_on arch: :arm64
  depends_on macos: :tahoe

  app "RawAlchemy.app"

  caveats <<~EOS
    Launch the local graphical application with:
      open -a RawAlchemy

    This release is not notarized. If macOS blocks the first launch,
    open System Settings > Privacy & Security, click Open Anyway for
    RawAlchemy, then confirm Open. Apple's instructions:
      https://support.apple.com/102445
  EOS
end
