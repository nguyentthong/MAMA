import argparse
from pytube import YouTube


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-url", type=str)
    parser.add_argument("--save-file", type=str)
    args = parser.parse_args()

    youtube = YouTube(args.video_url)
    youtube.streams.filter(file_extension='mp4')
    youtube.streams.get_by_itag(18).download(filename=args.save_file)


if __name__ == '__main__':
    main()