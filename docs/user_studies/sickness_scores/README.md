# Exclusive OR function experiment based on sickness score

```shell
npm install
npm build
```

Using a python 2 environment

```
pip install psiturk
```

Create a `~/.psiturkconfig` with your credentials:

```
[AWS Access]
aws_access_key_id = XXXXXXXXXXXXXX
aws_secret_access_key = XXXXXXXXXXXXXX
aws_region = us-east-1

[psiTurk Access]
psiturk_access_key_id = XXXXXXXXXXXXXX
psiturk_secret_access_id = XXXXXXXXXXXXXX
```

In `config.txt` replace the host name with the public name of the machine.

In the current directory run PsiTurk

```
psiturk

> server on
> debug
```

Now you can test things and download results with
```
> download_datafiles
```
