# Script to be run after transitions have been generated for a song.
# Having to do a bunch of crazy stuff sugggested by copilot to exit the script if a python script fails
# exit 1 doesn't work because it terminates the terminal

#!/bin/bash

# Function to stop the script execution without terminating the terminal
function stop_script {
  echo "$1 failed"
  stop=1
}

# Initialize a flag
stop=0


# Wrap your script inside a function
function main {

    # Initialize the variables
    song=$1
    scene_sequence=""
    output_name=""

    # Parse the arguments
    while (( "$#" )); do
    case "$1" in
        --ss)
        scene_sequence="$2"
        shift 2
        ;;
        -o)
        output_name="$2"
        shift 2
        ;;
        --) # end argument parsing
        shift
        break
        ;;
        -*|--*=) # unsupported flags
        echo "Error: Unsupported flag $1" >&2
        exit 1
        ;;
        *) # preserve positional arguments
        PARAMS="$PARAMS $1"
        shift
        ;;
    esac
    done
    # set positional arguments in their proper place
    eval set -- "$PARAMS"
    
    # Run the Python scripts and stop if any of them fail
    if [ -n "$scene_sequence" ]; then
        python examine_existing.py $song --ss $scene_sequence || stop_script 'examine_existing.py'
        [ "$stop" -eq 1 ] && return
        python story_nx_sections.py $song --ss $scene_sequence || stop_script 'story_nx_sections.py'
        [ "$stop" -eq 1 ] && return
    else
        python examine_existing.py $song || stop_script 'examine_existing.py'
        [ "$stop" -eq 1 ] && return
        python story_nx_sections.py $song || stop_script 'story_nx_sections.py'
        [ "$stop" -eq 1 ] && return
    fi

    if [ -n "$output_name" ]; then
        python gen_movie.py $song -o $output_name || stop_script 'gen_movie.py'
        [ "$stop" -eq 1 ] && return
    else
        python gen_movie.py $song || stop_script 'gen_movie.py'
        [ "$stop" -eq 1 ] && return
    fi

    # Can use this to test stopping the script
    # stop_script "test"

}
# Call the main function with all command line arguments
main "$@"