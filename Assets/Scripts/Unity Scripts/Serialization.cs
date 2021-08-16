using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;
using Newtonsoft.Json;

public static class Serialization {

    [Serializable]
    public class ImageObject {
        public string c2wM;
        public string projM;
        public List<byte> imageData;
        public string imageDataBase64;
        
        public ImageObject(Matrix4x4 c2wM, Matrix4x4 projM, List<byte> imgData) {
            this.c2wM = MatrixToString(c2wM);
            this.projM = MatrixToString(projM);
            this.imageData = imgData;
            this.imageDataBase64 = Convert.ToBase64String(imgData.ToArray());
        }

        private string MatrixToString(Matrix4x4 m) {
            string mString = "";
            int iEl = 0;

            for (int iRow = 0; iRow < 4; iRow++) {
                for (int iCol = 0; iCol < 4; iCol++) {
                    if (iEl > 0) {
                        mString += ",";
                    }

                    mString += string.Format("{0:F5}", m[iRow, iCol]);
                    iEl++;
                }
            }
            return mString;
        }

        public override string ToString() {
            return this.c2wM + ';' + this.projM + ';' + imageDataBase64;
        }
    }

    
    [Serializable]
    public class ComputePositionRequest {
        public string imageBufferFile;
        public string imageMetaFile;
        public int numPanels;
        public UITool.PanelConstraints[] constraints;
        public bool occlusion;
        public bool colorHarmony;
        public float colorfulness;
        public float edgeness;
        public float fittsLaw;
        public float ce;
        public float muscleActivation;
        public float rula;

        public ComputePositionRequest(string imageBufferFile, string imageMetaFile, int numPanels, UITool.PanelConstraints[] constraints, bool occlusion,
                                      bool colorHarmony, float colorfulness, float edgeness, float fittsLaw,
                                      float ce, float muscleActivation, float rula) {
            this.imageBufferFile = imageBufferFile;
            this.imageMetaFile = imageMetaFile;
            this.numPanels = numPanels;
            this.constraints = constraints;
            this.occlusion = occlusion;
            this.colorHarmony = colorHarmony;
            this.colorfulness = colorfulness;
            this.edgeness = edgeness;
            this.fittsLaw = fittsLaw;
            this.ce = ce;
            this.muscleActivation = muscleActivation;
            this.rula = rula;
        }
    }
}
