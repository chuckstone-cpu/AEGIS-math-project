"""Compose the final demo video: Webots 3D view | live engine dashboard.

    python webots/tools/compose_video.py <run_dir> <ffmpeg.exe> [dash_fps]
Both streams are on simulation time: the movie is recorded at 1x sim time and
the controller writes one dashboard JPEG every 6 control steps (6 x 16 ms).
"""
import glob
import os
import subprocess
import sys


def main():
    run, ffmpeg = sys.argv[1], sys.argv[2]
    dash_fps = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0 / (6 * 0.016)
    movies = [m for m in glob.glob(os.path.join(run, "*.mp4"))
              if not m.endswith("_final.mp4")]
    if not movies:
        sys.exit("no Webots movie in " + run)
    movie = movies[0]
    name = os.path.splitext(os.path.basename(movie))[0]
    out = os.path.join(run, name + "_final.mp4")
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", movie,
        "-framerate", f"{dash_fps:.4f}", "-i", os.path.join(run, "dash", "%06d.jpg"),
        "-filter_complex",
        "[0:v]scale=1280:720,setsar=1,fps=30[a];"
        "[1:v]scale=1280:720,setsar=1,fps=30[b];"
        "[a][b]hstack=inputs=2,format=yuv420p[v]",
        "-map", "[v]", "-shortest", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        out,
    ]
    subprocess.run(cmd, check=True)
    print("wrote", out, f"({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
