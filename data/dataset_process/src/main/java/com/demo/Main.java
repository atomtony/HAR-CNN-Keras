package com.demo;

import org.apache.commons.io.FileUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Collection;
import java.util.Date;
import java.util.Iterator;
import java.util.List;

public class Main {

    private static final SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");


    private static void dealFile(File csvFile, BufferedWriter writer, String userId, String activity) throws IOException, ParseException {
        List<String> lines = FileUtils.readLines(csvFile, Charset.defaultCharset());

        for (String line : lines) {
            // 33,Jogging,49106442306000,1.3756552,-2.4925237,-6.510526
            if (!line.contains("X") && line.length() > 0) {
                String[] array = line.split(",");
                Date date = formatter.parse(array[0]);
                long millisTimestamp = date.getTime() + Integer.parseInt(array[1]);
                StringBuilder builder = new StringBuilder()
                        .append(userId).append(",")
                        .append(activity).append(",")
                        .append(millisTimestamp).append(",")
                        .append(array[2]).append(",")
                        .append(array[3]).append(",")
                        .append(array[4]).append("\n");
                writer.write(builder.toString());
            }
        }
    }

    public static void main(String[] args) throws IOException, ParseException {
        File datasetDir = new File(args[0]);
        if (datasetDir.isDirectory()) {
            Collection<File> files = FileUtils.listFiles(datasetDir, new String[]{"csv"}, true);
            Iterator<File> iterator = files.iterator();


            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(datasetDir.getAbsolutePath(), "merged.txt")));
            while (iterator.hasNext()) {
                File file = iterator.next();
                String[] paths = file.getAbsolutePath().split("/");
                String userId = paths[paths.length - 1 - 2];
                String activity = paths[paths.length - 1 - 1];
                dealFile(file, writer, userId, activity);
            }
            writer.flush();
            writer.close();
        } else {
            String destFilename = datasetDir.getName().split("\\.")[0] + ".txt";
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(datasetDir.getParent(), destFilename)));
            String[] array = datasetDir.getName().split("_");
            dealFile(datasetDir, writer, array[0], array[1]);
            writer.flush();
            writer.close();
        }

    }
}
