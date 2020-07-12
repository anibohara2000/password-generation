# Metrics

The metrics used in this project are described in the [report](../RnD_Report.pdf). The code for the same is contained here. Following are the instructions to run the metrics.

## Memorability

To get the memorability, run the following commands in the folder:

```
python3 memorability.py --pwd=Password % Check memorability score of "Password"
python3 memorability.py --file=filename % Check memorability score of passwords contained in the file filename
```

## Guessability

To run the code to generate guessability scores, apache2 server is required. Install it using the following commands:

```
sudo apt-get update
sudo apt-get instal apache2
```

Then move files from the folder `copy_to_apache/` to `/var/www/html/` using the following command:

```
sudo cp copy_to_apache/* /var/www/html/.
```

Start the apache server:

```
sudo service apache2 start
```

To get the guessability, run the following commands in the folder:

```
python3 guessability.py --pwd=Password % Check guessability score of "Password"
python3 guessability.py --file=filename % Check guessability score of passwords contained in the file filename
```

After running the code for all files, stop the apache server using the command:

```
sudo service apache2 stop
```