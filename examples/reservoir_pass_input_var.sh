for f
do
    b="${f%.*}"
    gnuplot -e "INPUTFILE='$f'; OUTPUTFILE='${b}.png'" reservoir_pass_input_var.gpt
done


