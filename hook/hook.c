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

static unsigned char FRAME_BUFFER[768 * 480 * 3];

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

  char *move = zstr_recv(req_sock);

  zstr_free(&move);
  zsock_destroy(&req_sock);

  lib_glClear(mask);
}
