#!/bin/bash

dmg_name=ParaView-4.2.0-Darwin-64bit.dmg
vol_name=ParaView-4.2.0-Darwin-64bit

tmp_dir=reallylongpaththatisabout130characterslongthatshowsthatyoucanbreakdylibloadingwhenusingexecutablepathman130charactersislong

#get the current working directory
path=$(pwd)

#download the paraview dmg if we don't already have it
if ! [ -f "${path}/${dmg_name}" ]; then
  wget http://www.paraview.org/files/v4.2/ParaView-4.2.0-Darwin-64bit.dmg
fi

#open the dmg since it currently isn't mounted
if ! [ -d "/Volumes/${vol_name}" ]; then
  hdiutil attach -noverify "${path}/${dmg_name}"
  printf "Waiting for ParaView to Mount \n"
fi

while ! [ -d "/Volumes/${vol_name}" ]; do
  # if the volume hasn't mounted yet wait for it too
  printf "...\n"
  sleep 5
done

if ! [ -d "/tmp/${tmp_dir}/paraview.app" ]; then
  printf "Copying Paraview to directory structure that breaks dylib loading \n"
  mkdir -p "/tmp/${tmp_dir}/paraview.app"
  #copy the directory over
  cp -R "/Volumes/${vol_name}/paraview.app" "/tmp/${tmp_dir}/"
fi

if [ -d "/Volumes/${vol_name}" ]; then
  #unmount the paraview volume
  hdiutil detach "/Volumes/${vol_name}"
fi

#now try to launch paraview
open "/tmp/${tmp_dir}/paraview.app"