#define _GNU_SOURCE

#include <GL/gl.h>
#include <GL/glx.h>

#include <zmq.h>
#include <czmq.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

static unsigned char FRAME_BUFFER[768 * 480 * 3];

time_t PREV_SEC = 0;
suseconds_t PREV_USEC = 0;
int READY_FOR_NEXT_FRAME = 0;

// ~ 1000000 / 60 / 2
static suseconds_t TV_USEC_INCREMENT = 8300;

/**
 * We also need to hook `gettimeofday` because the game tries to lock its game
 * logic to the framerate of 60FPS, and our model may take longer to evaluate.
 *
 * We can trick the game into rendering a frame for any game-time interval by
 * adjusting TV_USEC_INCREMENT.
 */
int gettimeofday(struct timeval *tv, struct timezone *tz)
{
  static int (*lib_gettimeofday)(struct timeval *tv, struct timezone *tz) = NULL;
  lib_gettimeofday = dlsym(RTLD_NEXT, "gettimeofday");

  int ret = lib_gettimeofday(tv, tz);

  if (READY_FOR_NEXT_FRAME) {
    PREV_USEC = (PREV_USEC + TV_USEC_INCREMENT) % 1000000;
    tv->tv_usec = PREV_USEC;
    if (tv->tv_usec < TV_USEC_INCREMENT) {
      PREV_SEC++;
    }
    tv->tv_sec = PREV_SEC;
    READY_FOR_NEXT_FRAME = 0;
  } else {
    tv->tv_sec = PREV_SEC;
    tv->tv_usec = PREV_USEC;
  }

  return ret;
}

void glClear(GLbitfield mask)
{
  static void (*lib_glClear)(GLbitfield mask) = NULL;
  lib_glClear = dlsym(RTLD_NEXT, "glClear");

  // Read the frame data into a buffer

  glReadBuffer(GL_FRONT);
  glReadPixels(0, 0, 768, 480, GL_RGB, GL_UNSIGNED_BYTE, FRAME_BUFFER);

  // Create a zmq message
  zmsg_t *msg = zmsg_new();
  zframe_t *frame = zframe_new(FRAME_BUFFER, 768 * 480 * 3);

  zmsg_prepend(msg, &frame);

  // Open up the socket so we can send the frame to Python

  zsock_t *req_sock = zsock_new_req("tcp://localhost:5555");
  zmsg_send(&msg, req_sock);

  // The response is just used as an ack, so we just throw it away.

  char *move = zstr_recv(req_sock);
  READY_FOR_NEXT_FRAME = 1;

  zstr_free(&move);
  zsock_destroy(&req_sock);

  lib_glClear(mask);
}
