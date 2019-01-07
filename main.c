#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include <libavutil/bswap.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>

#include "libjpeg-turbo/turbojpeg.h"

void DSPCorrelateCoefs(const short* source, int samples, short coefsOut[8][2]);
void DSPEncodeFrame(short pcmInOut[16], int sampleCount, unsigned char adpcmOut[8], const short coefsIn[8][2]);

struct THPHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t maxBufferSize;
  uint32_t maxAudioSamples;
  float fps;
  uint32_t numFrames;
  uint32_t firstFrameSize;
  uint32_t dataSize;
  uint32_t componentDataOffset;
  uint32_t offsetsDataOffset;
  uint32_t firstFrameOffset;
  uint32_t lastFrameOffset;
};

static void THPHeader_swap(struct THPHeader* h) {
  h->magic = av_be2ne32(h->magic);
  h->version = av_be2ne32(h->version);
  h->maxBufferSize = av_be2ne32(h->maxBufferSize);
  h->maxAudioSamples = av_be2ne32(h->maxAudioSamples);
  *((uint32_t*)&h->fps) = av_be2ne32(*((uint32_t*)&h->fps));
  h->numFrames = av_be2ne32(h->numFrames);
  h->firstFrameSize = av_be2ne32(h->firstFrameSize);
  h->dataSize = av_be2ne32(h->dataSize);
  h->componentDataOffset = av_be2ne32(h->componentDataOffset);
  h->offsetsDataOffset = av_be2ne32(h->offsetsDataOffset);
  h->firstFrameOffset = av_be2ne32(h->firstFrameOffset);
  h->lastFrameOffset = av_be2ne32(h->lastFrameOffset);
}

enum THPComponent {
  THPVideo = 0, THPAudio = 1, THPNone = 0xff
};

struct THPComponents {
  uint32_t numComponents;
  uint8_t comps[16];
};

static void THPComponents_swap(struct THPComponents* c) {
  c->numComponents = av_be2ne32(c->numComponents);
}

struct THPVideoInfo {
  uint32_t width;
  uint32_t height;
};

static void THPVideoInfo_swap(struct THPVideoInfo* v) {
  v->width = av_be2ne32(v->width);
  v->height = av_be2ne32(v->height);
}

struct THPAudioInfo {
  uint32_t numChannels;
  uint32_t sampleRate;
  uint32_t numSamples;
};

static void THPAudioInfo_swap(struct THPAudioInfo* a) {
  a->numChannels = av_be2ne32(a->numChannels);
  a->sampleRate = av_be2ne32(a->sampleRate);
  a->numSamples = av_be2ne32(a->numSamples);
}

struct THPFrameHeader {
  uint32_t nextSize;
  uint32_t prevSize;
  uint32_t imageSize;
  uint32_t audioSize;
};

static void THPFrameHeader_swap(struct THPFrameHeader* f) {
  f->nextSize = av_be2ne32(f->nextSize);
  f->prevSize = av_be2ne32(f->prevSize);
  f->imageSize = av_be2ne32(f->imageSize);
  f->audioSize = av_be2ne32(f->audioSize);
}

struct THPAudioFrameHeader {
  uint32_t channelSize;
  uint32_t numSamples;
  int16_t channelCoefs[2][8][2];
  int16_t channelPrevs[2][2];
};

static void THPAudioFrameHeader_swap(struct THPAudioFrameHeader* f) {
  f->channelSize = av_be2ne32(f->channelSize);
  f->numSamples = av_be2ne32(f->numSamples);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 8; ++j)
      for (int k = 0; k < 2; ++k)
        f->channelCoefs[i][j][k] = av_be2ne16(f->channelCoefs[i][j][k]);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      f->channelPrevs[i][j] = av_be2ne16(f->channelPrevs[i][j]);
}

static struct THPHeader thp_header = {};
static struct THPComponents thp_comps = {};
static struct THPVideoInfo thp_video_info = {};
static struct THPAudioInfo thp_audio_info = {};

static AVFormatContext *fmt_ctx = NULL;
static AVCodecContext *video_dec_ctx = NULL, *audio_dec_ctx;
static struct SwsContext *sws_ctx = NULL;
static struct SwrContext *swr_ctx = NULL;
static tjhandle tj_handle = NULL;
static int jpeg_qual = 70;
static unsigned long jpeg_buf_size = 0;
static unsigned char *jpeg_buf = NULL;
static long int last_frame_off = 0;
static uint32_t last_frame_size = 0;
static int width, height;
static int out_width = 0, out_height = 0;
static enum AVPixelFormat pix_fmt;
static uint64_t channel_layout = 0;
static int out_srate = 0;
static double fps = 0.0;
static double samples_per_frame = 0.0;
static double current_sample = 0.0;
static uint32_t current_audio_packet = 0;
static AVStream *video_stream = NULL, *audio_stream = NULL;
static const char *src_filename = NULL;
static const char *dst_filename = NULL;
static FILE *dst_file = NULL;

static uint8_t *video_dst_data[4] = {NULL};
static int      video_dst_linesize[4];
static uint8_t *video_rgb_data[4] = {NULL};
static int      video_rgb_linesize[4];
static int video_dst_bufsize;

static int video_stream_idx = -1, audio_stream_idx = -1;
static AVFrame *frame = NULL;
static AVPacket pkt;
static int video_frame_count = 0;
static int audio_frame_count = 0;

static uint32_t audio_buf_capacity = 0;
static uint32_t audio_buf_size = 0;
static int16_t *audio_bufs[2] = {};

static uint8_t zero_buf[32] = {};

static int decode_packet(int *got_frame, int cached)
{
  int ret = 0;
  int decoded = pkt.size;

  *got_frame = 0;

  if (pkt.stream_index == video_stream_idx) {
    /* decode video frame */
    ret = avcodec_decode_video2(video_dec_ctx, frame, got_frame, &pkt);
    if (ret < 0) {
      fprintf(stderr, "Error decoding video frame (%s)\n", av_err2str(ret));
      return ret;
    }

    if (*got_frame) {

      if (frame->width != width || frame->height != height ||
          frame->format != pix_fmt) {
        /* To handle this change, one could call av_image_alloc again and
         * decode the following frames into another rawvideo file. */
        fprintf(stderr, "Error: Width, height and pixel format have to be "
                        "constant in a rawvideo file, but the width, height or "
                        "pixel format of the input video changed:\n"
                        "old: width = %d, height = %d, format = %s\n"
                        "new: width = %d, height = %d, format = %s\n",
                width, height, av_get_pix_fmt_name(pix_fmt),
                frame->width, frame->height,
                av_get_pix_fmt_name(frame->format));
        return -1;
      }

      printf("video_frame%s n:%d coded_n:%d\n",
             cached ? "(cached)" : "",
             video_frame_count++, frame->coded_picture_number);

      /* copy decoded frame to destination buffer:
       * this is required since rawvideo expects non aligned data */
      av_image_copy(video_dst_data, video_dst_linesize,
                    (const uint8_t **)(frame->data), frame->linesize,
                    pix_fmt, width, height);

      /* conversion to RGB is required */
      if (sws_ctx)
        sws_scale(sws_ctx, video_dst_data, video_dst_linesize,
                  0, height, video_rgb_data, video_rgb_linesize);

      unsigned long jpeg_size;
      tjCompress2(tj_handle, video_rgb_data[0], out_width, video_rgb_linesize[0], out_height, TJPF_RGB,
                  &jpeg_buf, &jpeg_size, TJSAMP_420, jpeg_qual, TJFLAG_NOREALLOC);

      /* write to thp file */
      long int this_frame_off = ftell(dst_file);
      struct THPFrameHeader thp_frame_header = {};
      thp_frame_header.nextSize = thp_header.firstFrameSize;
      thp_frame_header.prevSize = last_frame_size;
      thp_frame_header.imageSize = (uint32_t)((jpeg_size + 3) & ~0x3);
      uint32_t audio_size = 0;
      if (audio_stream) {
        current_sample += samples_per_frame;
        uint32_t end_sample = (uint32_t)current_sample;
        uint32_t end_audio_packet = (end_sample + 13) / 14;
        uint32_t num_audio_packets = end_audio_packet - current_audio_packet;
        current_audio_packet = end_audio_packet;
        audio_size = sizeof(struct THPAudioFrameHeader) + num_audio_packets * 8 *
                     (channel_layout == AV_CH_LAYOUT_STEREO ? 2 : 1);
        thp_frame_header.audioSize = audio_size;
      }
      THPFrameHeader_swap(&thp_frame_header);
      fwrite(&thp_frame_header, 1, audio_stream ? 16 : 12, dst_file);
      fwrite(jpeg_buf, 1, jpeg_size, dst_file);
      unsigned long rem = jpeg_size % 4;
      if (rem)
        fwrite(zero_buf, 1, 4 - rem, dst_file);

      if (audio_stream) {
        for (int i = 0; i < audio_size / 32; ++i)
          fwrite(zero_buf, 1, 32, dst_file);
        fwrite(zero_buf, 1, audio_size % 32, dst_file);
      }

      /* 32-byte align frame */
      long int cur = ftell(dst_file);
      last_frame_size = (uint32_t)(cur - this_frame_off);
      uint32_t frame_size_rem = last_frame_size % 32;
      if (frame_size_rem) {
        fwrite(zero_buf, 1, 32 - frame_size_rem, dst_file);
        cur += 32 - frame_size_rem;
      }
      last_frame_size = (uint32_t)(cur - this_frame_off);

      thp_header.maxBufferSize = FFMAX(thp_header.maxBufferSize, last_frame_size);
      thp_header.numFrames += 1;
      if (thp_header.firstFrameSize == 0)
        thp_header.firstFrameSize = last_frame_size;
      thp_header.dataSize += last_frame_size;
      thp_header.lastFrameOffset = (uint32_t)this_frame_off;

      if (last_frame_off) {
        fseek(dst_file, last_frame_off, SEEK_SET);
        uint32_t next_size = av_be2ne32(last_frame_size);
        fwrite(&next_size, 1, 4, dst_file);
        fseek(dst_file, cur, SEEK_SET);
      }

      last_frame_off = this_frame_off;
    }
  } else if (pkt.stream_index == audio_stream_idx) {
    /* decode audio frame */
    ret = avcodec_decode_audio4(audio_dec_ctx, frame, got_frame, &pkt);
    if (ret < 0) {
      fprintf(stderr, "Error decoding audio frame (%s)\n", av_err2str(ret));
      return ret;
    }
    /* Some audio decoders decode only part of the packet, and have to be
     * called again with the remainder of the packet data.
     * Sample: fate-suite/lossless-audio/luckynight-partial.shn
     * Also, some decoders might over-read the packet. */
    decoded = FFMIN(ret, pkt.size);

    if (*got_frame) {
      printf("audio_frame%s n:%d nb_samples:%d pts:%s\n",
             cached ? "(cached)" : "",
             audio_frame_count++, frame->nb_samples,
             av_ts2timestr(frame->pts, &audio_dec_ctx->time_base));

      if (swr_ctx) {
        if (audio_buf_capacity - audio_buf_size < swr_get_out_samples(swr_ctx, frame->nb_samples)) {
          audio_buf_capacity += 10 * out_srate;
          audio_bufs[0] = realloc(audio_bufs[0], 2 * audio_buf_capacity);
          if (channel_layout == AV_CH_LAYOUT_STEREO)
            audio_bufs[1] = realloc(audio_bufs[1], 2 * audio_buf_capacity);
        }
        uint8_t* outs[2] = {(uint8_t*)(audio_bufs[0] + audio_buf_size),
                            (uint8_t*)(audio_bufs[1] + audio_buf_size)};
        int out_samples = swr_convert(swr_ctx, outs, audio_buf_capacity - audio_buf_size,
                                      (const uint8_t**)frame->extended_data, frame->nb_samples);
        audio_buf_size += out_samples;
      }
    }
  }

  /* If we use frame reference counting, we own the data and need
   * to de-reference it when we don't use it anymore */
  if (*got_frame)
    av_frame_unref(frame);

  return decoded;
}

static int open_codec_context(int *stream_idx,
                              AVCodecContext **dec_ctx, AVFormatContext *fmt_ctx, enum AVMediaType type)
{
  int ret, stream_index;
  AVStream *st;
  AVCodec *dec = NULL;
  AVDictionary *opts = NULL;

  ret = av_find_best_stream(fmt_ctx, type, -1, -1, NULL, 0);
  if (ret < 0) {
    fprintf(stderr, "Could not find %s stream in input file '%s'\n",
            av_get_media_type_string(type), src_filename);
    return ret;
  } else {
    stream_index = ret;
    st = fmt_ctx->streams[stream_index];

    /* find decoder for the stream */
    dec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!dec) {
      fprintf(stderr, "Failed to find %s codec\n",
              av_get_media_type_string(type));
      return AVERROR(EINVAL);
    }

    /* Allocate a codec context for the decoder */
    *dec_ctx = avcodec_alloc_context3(dec);
    if (!*dec_ctx) {
      fprintf(stderr, "Failed to allocate the %s codec context\n",
              av_get_media_type_string(type));
      return AVERROR(ENOMEM);
    }

    /* Copy codec parameters from input stream to output codec context */
    if ((ret = avcodec_parameters_to_context(*dec_ctx, st->codecpar)) < 0) {
      fprintf(stderr, "Failed to copy %s codec parameters to decoder context\n",
              av_get_media_type_string(type));
      return ret;
    }

    /* Init the decoders, with or without reference counting */
    av_dict_set(&opts, "refcounted_frames", "1", 0);
    if ((ret = avcodec_open2(*dec_ctx, dec, &opts)) < 0) {
      fprintf(stderr, "Failed to open %s codec\n",
              av_get_media_type_string(type));
      return ret;
    }
    *stream_idx = stream_index;
  }

  return 0;
}

static int get_format_from_sample_fmt(const char **fmt,
                                      enum AVSampleFormat sample_fmt)
{
  int i;
  struct sample_fmt_entry {
    enum AVSampleFormat sample_fmt; const char *fmt_be, *fmt_le;
  } sample_fmt_entries[] = {
          { AV_SAMPLE_FMT_U8,  "u8",    "u8"    },
          { AV_SAMPLE_FMT_S16, "s16be", "s16le" },
          { AV_SAMPLE_FMT_S32, "s32be", "s32le" },
          { AV_SAMPLE_FMT_FLT, "f32be", "f32le" },
          { AV_SAMPLE_FMT_DBL, "f64be", "f64le" },
  };
  *fmt = NULL;

  for (i = 0; i < FF_ARRAY_ELEMS(sample_fmt_entries); i++) {
    struct sample_fmt_entry *entry = &sample_fmt_entries[i];
    if (sample_fmt == entry->sample_fmt) {
      *fmt = AV_NE(entry->fmt_be, entry->fmt_le);
      return 0;
    }
  }

  fprintf(stderr,
          "sample format %s is not supported as output format\n",
          av_get_sample_fmt_name(sample_fmt));
  return -1;
}

static void print_usage(char **argv) {
  fprintf(stderr, "usage: %s [-s WxH] [-q <1-100>] [-r <sample-rate>] <input-file> <thp-output-file>\n", argv[0]);
}

int main (int argc, char **argv)
{
  int ret = 0, got_frame;

  if (argc < 3) {
    print_usage(argv);
    exit(1);
  }

  for (int i = 1; i < argc - 3; ++i) {
    if (argv[i][0] == '-') {
      switch (argv[i][1]) {
      case 's':
        if (i + 1 >= argc || sscanf(argv[i + 1], "%ux%u", &out_width, &out_height) != 2) {
          fprintf(stderr, "Could not parse size argument\n");
          exit(1);
        }
        ++i;
        break;
      case 'q':
        if (i + 1 >= argc || sscanf(argv[i + 1], "%u", &jpeg_qual) != 1) {
          fprintf(stderr, "Could not parse quality argument\n");
          exit(1);
        }
        if (jpeg_qual < 1 || jpeg_qual > 100) {
          fprintf(stderr, "Quality argument out of range\n");
          exit(1);
        }
        ++i;
        break;
      case 'r':
        if (i + 1 >= argc || sscanf(argv[i + 1], "%u", &out_srate) != 1) {
          fprintf(stderr, "Could not sample rate argument\n");
          exit(1);
        }
        ++i;
        break;
      default:
        break;
      }
    }
  }

  src_filename = argv[argc - 2];
  dst_filename = argv[argc - 1];

  /* open input file, and allocate format context */
  if (avformat_open_input(&fmt_ctx, src_filename, NULL, NULL) < 0) {
    fprintf(stderr, "Could not open source file %s\n", src_filename);
    exit(1);
  }

  /* retrieve stream information */
  if (avformat_find_stream_info(fmt_ctx, NULL) < 0) {
    fprintf(stderr, "Could not find stream information\n");
    exit(1);
  }

  dst_file = fopen(dst_filename, "w+b");
  if (!dst_file) {
    fprintf(stderr, "Could not open destination file %s\n", dst_filename);
    ret = 1;
    goto end;
  }

  if (open_codec_context(&video_stream_idx, &video_dec_ctx, fmt_ctx, AVMEDIA_TYPE_VIDEO) >= 0) {
    video_stream = fmt_ctx->streams[video_stream_idx];

    /* allocate image where the decoded image will be put */
    width = video_dec_ctx->width;
    height = video_dec_ctx->height;
    pix_fmt = video_dec_ctx->pix_fmt;
    ret = av_image_alloc(video_dst_data, video_dst_linesize,
                         width, height, pix_fmt, 1);
    if (ret < 0) {
      fprintf(stderr, "Could not allocate raw video buffer\n");
      goto end;
    }
    video_dst_bufsize = ret;
  }

  if (open_codec_context(&audio_stream_idx, &audio_dec_ctx, fmt_ctx, AVMEDIA_TYPE_AUDIO) >= 0) {
    audio_stream = fmt_ctx->streams[audio_stream_idx];
  }

  /* dump input information to stderr */
  av_dump_format(fmt_ctx, 0, src_filename, 0);

  if (!video_stream) {
    fprintf(stderr, "Could not find audio or video stream in the input, aborting\n");
    ret = 1;
    goto end;
  }

  frame = av_frame_alloc();
  if (!frame) {
    fprintf(stderr, "Could not allocate frame\n");
    ret = AVERROR(ENOMEM);
    goto end;
  }

  if (out_width == 0)
    out_width = width;
  if (out_height == 0)
    out_height = height;

  /* non-RGB color format, setup sws to convert */
  if (pix_fmt != AV_PIX_FMT_RGB24) {
    sws_ctx = sws_getContext(width, height, pix_fmt, out_width, out_height,
                             AV_PIX_FMT_RGB24, SWS_BILINEAR, NULL, NULL, NULL);
    ret = av_image_alloc(video_rgb_data, video_rgb_linesize,
                         out_width, out_height, AV_PIX_FMT_RGB24, 1);
    if (ret < 0) {
      fprintf(stderr, "Could not allocate RGB video buffer\n");
      goto end;
    }
  } else {
    for (int i = 0; i < 4; ++i) {
      video_rgb_data[i] = video_dst_data[i];
      video_rgb_linesize[i] = video_dst_linesize[i];
    }
  }

  fps = video_stream->avg_frame_rate.num / (double)(video_stream->avg_frame_rate.den);

  if (out_srate == 0)
    out_srate = audio_dec_ctx->sample_rate;

  if (audio_stream) {
    channel_layout = (audio_dec_ctx->channel_layout != AV_CH_LAYOUT_MONO &&
                      audio_dec_ctx->channel_layout != AV_CH_LAYOUT_STEREO) ?
                          AV_CH_LAYOUT_STEREO : audio_dec_ctx->channel_layout;
    swr_ctx = swr_alloc_set_opts(NULL, channel_layout, AV_SAMPLE_FMT_S16P, out_srate,
              audio_dec_ctx->channel_layout, audio_dec_ctx->sample_fmt, audio_dec_ctx->sample_rate, 0, NULL);
    swr_init(swr_ctx);
    /* Preallocate 10-seconds of audio by default */
    audio_buf_capacity = 10 * out_srate;
    audio_bufs[0] = malloc(2 * audio_buf_capacity);
    if (channel_layout == AV_CH_LAYOUT_STEREO)
      audio_bufs[1] = malloc(2 * audio_buf_capacity);

    samples_per_frame = out_srate / fps;
  }


  /* initialize turbojpeg */
  tj_handle = tjInitCompress();
  if (!tj_handle) {
    fprintf(stderr, "Could not initialize turbojpeg\n");
    goto end;
  }

  jpeg_buf_size = tjBufSize(out_width, out_height, TJSAMP_420);
  jpeg_buf = tjAlloc(jpeg_buf_size);
  if (!jpeg_buf) {
    fprintf(stderr, "Could not allocate JPEG buffer\n");
    goto end;
  }

  /* setup thp header */
  thp_header.magic = 'THP\0';
  thp_header.version = 0x00010000;
  thp_header.fps = (float)fps;
  thp_header.componentDataOffset = sizeof(struct THPHeader);
  thp_header.firstFrameOffset = thp_header.componentDataOffset +
          sizeof(struct THPComponents) + sizeof(struct THPVideoInfo);
  memset(thp_comps.comps, 0xff, 16);
  thp_comps.numComponents = 1;
  thp_comps.comps[0] = THPVideo;
  thp_video_info.width = (uint32_t)out_width;
  thp_video_info.height = (uint32_t)out_height;
  if (audio_stream) {
    thp_header.firstFrameOffset += sizeof(struct THPAudioInfo);
    thp_comps.numComponents = 2;
    thp_comps.comps[1] = THPAudio;
  }

  /* seek to first frame */
  fseek(dst_file, thp_header.firstFrameOffset, SEEK_SET);

  /* initialize packet, set data to NULL, let the demuxer fill it */
  av_init_packet(&pkt);
  pkt.data = NULL;
  pkt.size = 0;

  if (video_stream)
    printf("Demuxing video from file '%s' into '%s'\n", src_filename, dst_filename);
  if (audio_stream)
    printf("Demuxing audio from file '%s' into '%s'\n", src_filename, dst_filename);

  /* read frames from the file */
  while (av_read_frame(fmt_ctx, &pkt) >= 0) {
    AVPacket orig_pkt = pkt;
    do {
      ret = decode_packet(&got_frame, 0);
      if (ret < 0)
        break;
      pkt.data += ret;
      pkt.size -= ret;
    } while (pkt.size > 0);
    av_packet_unref(&orig_pkt);
  }

  /* flush cached frames */
  pkt.data = NULL;
  pkt.size = 0;
  do {
    decode_packet(&got_frame, 1);
  } while (got_frame);

  fseek(dst_file, thp_header.firstFrameOffset + 4, SEEK_SET);
  uint32_t prev_size = av_be2ne32(last_frame_size);
  fwrite(&prev_size, 1, 4, dst_file);

  /* audio pass */
  if (audio_stream) {
    printf("Compressing audio...\n");
    uint32_t channel_count = 1;
    if (channel_layout == AV_CH_LAYOUT_STEREO)
      channel_count = 2;
    thp_audio_info.numChannels = channel_count;
    thp_audio_info.sampleRate = (uint32_t)out_srate;
    thp_audio_info.numSamples = audio_buf_size;

    int16_t *sample_cur[2] = {audio_bufs[0], audio_bufs[1]};
    fseek(dst_file, thp_header.firstFrameOffset, SEEK_SET);
    uint32_t this_size = thp_header.firstFrameSize;
    int16_t conv_samps[2][16] = {};
    uint32_t full_rem_samples = audio_buf_size;
    for (uint32_t f = 0; f < thp_header.numFrames; ++f) {
      long int next_off = ftell(dst_file) + this_size;
      uint32_t next_size, image_size, audio_size;
      fread(&next_size, 1, 4, dst_file);
      next_size = av_be2ne32(next_size);
      fseek(dst_file, 4, SEEK_CUR);
      fread(&image_size, 1, 4, dst_file);
      image_size = av_be2ne32(image_size);
      fread(&audio_size, 1, 4, dst_file);
      audio_size = av_be2ne32(audio_size);
      fseek(dst_file, image_size, SEEK_CUR);
      uint32_t num_audio_packets = (audio_size - sizeof(struct THPAudioFrameHeader)) /
              (8 * (channel_layout == AV_CH_LAYOUT_STEREO ? 2 : 1));

      struct THPAudioFrameHeader audio_header = {};
      if (channel_layout == AV_CH_LAYOUT_STEREO)
        audio_header.channelSize = num_audio_packets * 8;
      uint32_t num_samples = FFMIN(num_audio_packets * 14, full_rem_samples);
      uint32_t rem_samples[2] = {num_samples, num_samples};
      audio_header.numSamples = num_samples;
      thp_header.maxAudioSamples = FFMAX(thp_header.maxAudioSamples, num_samples);

      for (uint32_t c = 0; c < channel_count; ++c) {
        DSPCorrelateCoefs(sample_cur[c], num_samples, audio_header.channelCoefs[c]);
        audio_header.channelPrevs[c][0] = conv_samps[c][14];
        audio_header.channelPrevs[c][1] = conv_samps[c][15];
      }

      THPAudioFrameHeader_swap(&audio_header);
      fwrite(&audio_header, 1, sizeof(struct THPAudioFrameHeader), dst_file);
      THPAudioFrameHeader_swap(&audio_header);

      for (uint32_t c = 0; c < channel_count; ++c) {
        for (uint32_t i = 0; i < num_audio_packets; ++i) {
          uint32_t sample_count = FFMIN(14u, rem_samples[c]);
          conv_samps[c][0] = conv_samps[c][14];
          conv_samps[c][1] = conv_samps[c][15];
          memcpy(conv_samps[c] + 2, sample_cur[c], sample_count * 2);
          sample_cur[c] += sample_count;
          unsigned char adpcm_out[8];
          DSPEncodeFrame(conv_samps[c], sample_count, adpcm_out, audio_header.channelCoefs[c]);
          fwrite(adpcm_out, 1, 8, dst_file);
          rem_samples[c] -= sample_count;
        }
      }

      full_rem_samples -= num_samples;

      fseek(dst_file, next_off, SEEK_SET);
      this_size = next_size;
    }
  }

  fseek(dst_file, 0, SEEK_SET);
  THPHeader_swap(&thp_header);
  fwrite(&thp_header, 1, sizeof(struct THPHeader), dst_file);
  THPComponents_swap(&thp_comps);
  fwrite(&thp_comps, 1, sizeof(struct THPComponents), dst_file);
  THPVideoInfo_swap(&thp_video_info);
  fwrite(&thp_video_info, 1, sizeof(struct THPVideoInfo), dst_file);
  if (audio_stream) {
    THPAudioInfo_swap(&thp_audio_info);
    fwrite(&thp_audio_info, 1, sizeof(struct THPAudioInfo), dst_file);
  }

  printf("THP conversion succeeded.\n");

  if (out_width > 720 || out_height > 480)
    fprintf(stderr, "WARNING: Potentially excessive video dimensions.\n"
                    "Consider using `-s 640x480` for a sensible value.\n");
  if (out_width & 0xf || out_height & 0xf)
    fprintf(stderr, "WARNING: One dimension not a multiple of 16.\n"
                    "Consider using `-s 640x480` for a sensible value.\n");
  if (out_srate > 48000)
    fprintf(stderr, "WARNING: Potentially excessive sample rate.\n"
                    "Consider using `-r 32000` for a sensible value.\n");

end:
  if (jpeg_buf)
    tjFree(jpeg_buf);
  if (tj_handle)
    tjDestroy(tj_handle);
  if (swr_ctx)
    swr_free(&swr_ctx);
  if (sws_ctx)
    sws_freeContext(sws_ctx);
  avcodec_free_context(&video_dec_ctx);
  avcodec_free_context(&audio_dec_ctx);
  avformat_close_input(&fmt_ctx);
  if (dst_file)
    fclose(dst_file);
  av_frame_free(&frame);
  av_free(video_dst_data[0]);
  if (audio_bufs[0])
    free(audio_bufs[0]);
  if (audio_bufs[1])
    free(audio_bufs[1]);

  return ret < 0;
}
