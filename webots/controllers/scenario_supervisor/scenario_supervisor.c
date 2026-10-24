/*
 * scenario_supervisor - drives the APEX demo inside Webots.
 *
 *  - records the 3D view to an MP4 (--movie file.mp4, --width/--height)
 *  - quits the simulation after --duration seconds (0 = never)
 *  - "stop & go": every node whose DEF name starts with "OBST_" is a blocking
 *    obstacle. When the ego car (DEF EGO) has been stopped behind one for
 *    --clear-after seconds, that obstacle is removed (the stalled car drives
 *    away / the cones are cleared), so one run shows many scenarios in turn.
 *  - kinematic actors (--actors file): moving cars that follow a polyline at
 *    a fixed speed once the ego car comes within a trigger radius. Used for a
 *    slow lead vehicle and a cut-in. Format, one actor per line:
 *        DEF trigger_x trigger_y trigger_radius speed_kmh x1 y1 x2 y2 ...
 *  - writes ego pose + events to --log file.csv
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <webots/robot.h>
#include <webots/supervisor.h>

#define MAX_OBST 64
#define MAX_ACTORS 8
#define MAX_WP 256

typedef struct {
  WbNodeRef node;
  char name[64];
  int alive;
  int actor; /* index into actors[] or -1 */
} Obstacle;

typedef struct {
  char def[64];
  double tx, ty, tr, speed; /* trigger point, radius [m], speed [m/s] */
  int n;
  double x[MAX_WP], y[MAX_WP], s[MAX_WP]; /* polyline + arc length */
  WbNodeRef node;
  int state; /* 0 waiting, 1 moving, 2 done */
  double dist;
} Actor;

static const char *arg_value(int argc, char **argv, const char *key, const char *def) {
  for (int i = 1; i < argc - 1; ++i)
    if (strcmp(argv[i], key) == 0)
      return argv[i + 1];
  return def;
}

static int load_actors(const char *path, Actor *a) {
  FILE *f = fopen(path, "r");
  if (!f)
    return 0;
  char line[8192];
  int n = 0;
  while (n < MAX_ACTORS && fgets(line, sizeof(line), f)) {
    if (line[0] == '#' || strlen(line) < 5)
      continue;
    Actor *ac = &a[n];
    memset(ac, 0, sizeof(*ac));
    char *p = line;
    int used = 0;
    double kmh;
    if (sscanf(p, "%63s %lf %lf %lf %lf%n", ac->def, &ac->tx, &ac->ty, &ac->tr, &kmh, &used) != 5)
      continue;
    ac->speed = kmh / 3.6;
    p += used;
    while (ac->n < MAX_WP && sscanf(p, "%lf %lf%n", &ac->x[ac->n], &ac->y[ac->n], &used) == 2) {
      p += used;
      ac->s[ac->n] = ac->n ? ac->s[ac->n - 1] + hypot(ac->x[ac->n] - ac->x[ac->n - 1],
                                                     ac->y[ac->n] - ac->y[ac->n - 1])
                           : 0.0;
      ac->n++;
    }
    if (ac->n >= 2)
      ++n;
  }
  fclose(f);
  return n;
}

/* position + heading at arc length d along the actor polyline */
static void actor_pose(const Actor *ac, double d, double *x, double *y, double *yaw) {
  int k = 0;
  while (k < ac->n - 2 && ac->s[k + 1] < d)
    ++k;
  const double seg = ac->s[k + 1] - ac->s[k];
  const double u = seg > 1e-9 ? fmin(1.0, fmax(0.0, (d - ac->s[k]) / seg)) : 0.0;
  *x = ac->x[k] + u * (ac->x[k + 1] - ac->x[k]);
  *y = ac->y[k] + u * (ac->y[k + 1] - ac->y[k]);
  *yaw = atan2(ac->y[k + 1] - ac->y[k], ac->x[k + 1] - ac->x[k]);
}

int main(int argc, char **argv) {
  wb_robot_init();
  const int step = (int)wb_robot_get_basic_time_step();
  const double dt = step / 1000.0;
  const char *movie = arg_value(argc, argv, "--movie", "");
  const double duration = atof(arg_value(argc, argv, "--duration", "0"));
  const double clear_after = atof(arg_value(argc, argv, "--clear-after", "4"));
  const int mw = atoi(arg_value(argc, argv, "--width", "1280"));
  const int mh = atoi(arg_value(argc, argv, "--height", "720"));
  const char *log_path = arg_value(argc, argv, "--log", "");
  FILE *log = log_path[0] ? fopen(log_path, "w") : NULL;
  if (log)
    fprintf(log, "t,x,y,yaw,speed_kmh,event\n");

  WbNodeRef ego = wb_supervisor_node_get_from_def("EGO");
  if (!ego) {
    fprintf(stderr, "[supervisor] no node DEF EGO\n");
    wb_robot_cleanup();
    return 1;
  }

  static Actor actors[MAX_ACTORS];
  const int n_actors = load_actors(arg_value(argc, argv, "--actors", ""), actors);

  /* collect OBST_* nodes from the scene root */
  Obstacle obst[MAX_OBST];
  int n_obst = 0;
  WbFieldRef children = wb_supervisor_node_get_field(wb_supervisor_node_get_root(), "children");
  const int n_children = wb_supervisor_field_get_count(children);
  for (int i = 0; i < n_children && n_obst < MAX_OBST; ++i) {
    WbNodeRef n = wb_supervisor_field_get_mf_node(children, i);
    const char *def = wb_supervisor_node_get_def(n);
    if (def && strncmp(def, "OBST_", 5) == 0) {
      obst[n_obst].node = n;
      strncpy(obst[n_obst].name, def, 63);
      obst[n_obst].name[63] = 0;
      obst[n_obst].alive = 1;
      obst[n_obst].actor = -1;
      for (int k = 0; k < n_actors; ++k)
        if (strcmp(actors[k].def, def) == 0) {
          actors[k].node = n;
          obst[n_obst].actor = k;
        }
      ++n_obst;
    }
  }
  printf("[supervisor] %d obstacles, %d moving actors, duration %.0f s\n", n_obst, n_actors,
         duration);
  fflush(stdout);

  int recording = 0;
  double stopped_since = -1.0, last_log = -1.0;
  while (wb_robot_step(step) != -1) {
    const double t = wb_robot_get_time();
    if (!recording && movie[0] && t > 0.1) {
      wb_supervisor_movie_start_recording(movie, mw, mh, 0, 90, 1, false);
      recording = 1;
    }

    const double *p = wb_supervisor_node_get_position(ego);
    const double *R = wb_supervisor_node_get_orientation(ego);
    const double *vel = wb_supervisor_node_get_velocity(ego);
    const double yaw = atan2(R[3], R[0]);
    const double speed = sqrt(vel[0] * vel[0] + vel[1] * vel[1]) * 3.6;
    char event[128] = "";

    /* kinematic actors */
    for (int k = 0; k < n_actors; ++k) {
      Actor *ac = &actors[k];
      if (!ac->node || ac->state == 2)
        continue;
      if (ac->state == 0 && hypot(p[0] - ac->tx, p[1] - ac->ty) < ac->tr) {
        ac->state = 1;
        snprintf(event, sizeof(event), "start %s", ac->def);
        printf("[supervisor] t=%.1f actor %s starts\n", t, ac->def);
      }
      if (ac->state == 1) {
        ac->dist += ac->speed * dt;
        if (ac->dist >= ac->s[ac->n - 1]) {
          ac->state = 2;
          for (int j = 0; j < n_obst; ++j)
            if (obst[j].actor == k && obst[j].alive) {
              wb_supervisor_node_remove(obst[j].node);
              obst[j].alive = 0;
            }
          snprintf(event, sizeof(event), "finish %s", ac->def);
          continue;
        }
        double x, y, a;
        actor_pose(ac, ac->dist, &x, &y, &a);
        const double tr[3] = {x, y, 0.4};
        const double rot[4] = {0, 0, 1, a};
        wb_supervisor_field_set_sf_vec3f(wb_supervisor_node_get_field(ac->node, "translation"), tr);
        wb_supervisor_field_set_sf_rotation(wb_supervisor_node_get_field(ac->node, "rotation"), rot);
        wb_supervisor_node_reset_physics(ac->node);
      }
    }

    /* stop & go: closest live STATIC obstacle in front of the ego car */
    int blocking = -1;
    double best = 1e9;
    for (int k = 0; k < n_obst; ++k) {
      if (!obst[k].alive || obst[k].actor >= 0)
        continue;
      const double *q = wb_supervisor_node_get_position(obst[k].node);
      const double dx = q[0] - p[0], dy = q[1] - p[1];
      const double ahead = dx * cos(yaw) + dy * sin(yaw);
      const double side = -dx * sin(yaw) + dy * cos(yaw);
      if (ahead > 0.0 && ahead < 18.0 && fabs(side) < 3.5 && ahead < best) {
        best = ahead;
        blocking = k;
      }
    }
    if (blocking >= 0 && speed < 1.0) {
      if (stopped_since < 0)
        stopped_since = t;
      if (t - stopped_since > clear_after) {
        printf("[supervisor] t=%.1f clearing %s (ego waited %.1f s, gap %.1f m)\n", t,
               obst[blocking].name, t - stopped_since, best);
        snprintf(event, sizeof(event), "clear %s", obst[blocking].name);
        wb_supervisor_node_remove(obst[blocking].node);
        obst[blocking].alive = 0;
        stopped_since = -1.0;
      }
    } else {
      stopped_since = -1.0;
    }

    if (log && (t - last_log >= 0.1 || event[0])) {
      fprintf(log, "%.2f,%.3f,%.3f,%.4f,%.2f,%s\n", t, p[0], p[1], yaw, speed, event);
      last_log = t;
    }

    if (duration > 0 && t >= duration) {
      if (recording) {
        wb_supervisor_movie_stop_recording();
        while (!wb_supervisor_movie_is_ready() && wb_robot_step(step) != -1) {
        }
        if (wb_supervisor_movie_failed())
          fprintf(stderr, "[supervisor] movie recording failed\n");
      }
      printf("[supervisor] done at t=%.1f\n", t);
      if (log)
        fclose(log);
      wb_supervisor_simulation_quit(0);
      wb_robot_step(step);
      break;
    }
  }
  wb_robot_cleanup();
  return 0;
}
