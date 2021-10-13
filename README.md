# Redpoll static website

This website is generated with [zola](https://www.getzola.org/). **Built with zola 0.14.1**

To install zola check out their [installation page](https://www.getzola.org/documentation/getting-started/installation/)

To launch the website on you local machine:

```
$ zola serve
```

## Development

All work should be done off the `draft` branch. The `master` branch contains
the live site.

To use bokeh plots in posts make sure `bokeh = true` in the `extras` section of
the post header. For specific use examples using the `jsplot` shortcode, see
`content/blog/sparse.md`

## Demos

Some blog posts may have demos inside them. Demo code should be included in the
`demos/` directory.

To ensure that any paths to data files are not pointing to places on an
individual user's file system, users have a `demos/paths.txt` file that is a
list of all files used by demos. Each row of the file is a file name followed
by `=` followed by the path. For example,

```
satellites = /Users/bax/redpoll/pybraid/examples/satellites/data.csv
animals = /Users/bax/redpoll/braid/resources/datasets/animals/data.csv
```

In python, you can get the path you need like so,

```python
with open('paths.txt') as f:
    paths = {}
    for line in f.readlines():
        filename, path = line.split('=')
        paths[filename.strip()] = path.strip()
        
path = paths['satellites']
```
