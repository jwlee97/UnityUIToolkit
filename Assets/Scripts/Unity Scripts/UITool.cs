using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
using Newtonsoft.Json;

public class UITool : MonoBehaviour {
    [Serializable]
    public struct PanelConstraints {
        public string name;
        public float height;
        public float width;
    }

    public Slider colorfulnessSlider;
    public Slider edgenessSlider;
    public Slider fittsLawSlider;
    public Slider ceSlider;
    public Slider muscleActivationSlider;
    public Slider rulaSlider;
    public Slider cognitiveLoadSlider;
    public Toggle enableOcclusion;
    public Toggle enableColorHarmony;

    public string imageBufferFile;
    public float[,] panelSizes;
    public PanelConstraints[] constraints;

    private GameObject[] panels;
    private List<string> panelData;
    private int numPanels;
    private bool UIGenerated;
   
    private PythonNetworking _pythonNetworking;
    private bool _clientStopped;
    private bool _requestPending;
    
    private GameObject labScene;
    private GameObject classScene;
    private GameObject officeScene;

    public void Start() {
        _pythonNetworking = new PythonNetworking(false);
        numPanels = constraints.Length;
        panelSizes = new float[numPanels, 2];

        if (imageBufferFile == null)
            imageBufferFile = "context_img_buff_1623145003.log";

        UIGenerated = false;
        labScene = GameObject.FindGameObjectWithTag("labScene");
        classScene = GameObject.FindGameObjectWithTag("classScene");
        officeScene = GameObject.FindGameObjectWithTag("officeScene");

        Debug.Log("UI tool initialized.");
        DeactivateScenes();
    }

    private void DeactivateScenes() {
        labScene.SetActive(false);
        classScene.SetActive(false);
        officeScene.SetActive(false);
    }

    public void SubmitConstraints() {
        StartCoroutine(CreateRequest());
        Debug.Log("Sending data to Python server.");
    }

    public void CreateUI() {
        int i = 0;
        
        if (panelData != null) {
            Debug.Log("Creating UI.");
            panels = new GameObject[numPanels];
            UIGenerated = true;

            if (panels != null)
                DestroyPanels();

            foreach (var line in panelData) {
                panels[i] = InitializePanel(line, i);
                i++;
            }

        } else {
            Debug.Log("Please wait: Python script still generating optimal UI.");
        }
    }
   
    private IEnumerator CreateRequest() {
        var request = new Serialization.ComputePositionRequest(imageBufferFile, numPanels, constraints, enableOcclusion.isOn, enableColorHarmony.isOn,
                                                               colorfulnessSlider.value, edgenessSlider.value, fittsLawSlider.value,
                                                               ceSlider.value, muscleActivationSlider.value, rulaSlider.value);
        var requestJson = JsonUtility.ToJson(request);
        _pythonNetworking.PerformRequest("C", requestJson);
        yield return new WaitUntil(() => _pythonNetworking.requestResult != null);
        panelData = JsonConvert.DeserializeObject<List<string>>(_pythonNetworking.requestResult);
        Debug.Log("Creating request, sending UI constraint to server.");
    }

    private void DestroyPanels() {
        GameObject[] panels = GameObject.FindGameObjectsWithTag("Panel");
        foreach(GameObject p in panels)
            GameObject.Destroy(p);
    }

    private GameObject InitializePanel(string data, int i) {
        string[] spl = data.Split(';');
        float height = float.Parse(spl[0].Split(',')[0]);
        float width = float.Parse(spl[0].Split(',')[1]);
        panelSizes[i, 0] = height;
        panelSizes[i, 1] = width;

        float[] panelPos = {float.Parse(spl[1].Split(',')[1]), float.Parse(spl[1].Split(',')[0]), float.Parse(spl[1].Split(',')[2])};
        Color panelColor = new Color(float.Parse(spl[2].Split(',')[0])/255, float.Parse(spl[2].Split(',')[1])/255, float.Parse(spl[2].Split(',')[2])/255);
        Color textColor = new Color(float.Parse(spl[3].Split(',')[0])/255, float.Parse(spl[3].Split(',')[1])/255, float.Parse(spl[3].Split(',')[2])/255);

        GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.tag = "Panel";
    
        cube.transform.position = new Vector3(panelPos[1], panelPos[0], 0.5f);
        cube.transform.localScale = new Vector3(width, height, 0.0001f);
        var cubeRenderer = cube.GetComponent<Renderer>();
        cubeRenderer.material.SetColor("_Color", panelColor);

        GameObject label = new GameObject();
        label.transform.parent = cube.transform;
        label.transform.localPosition = cube.transform.localPosition;
        label.transform.localScale = new Vector3(cube.transform.localScale.y, cube.transform.localScale.x, 1.0f);

        RectTransform rectTransform = label.AddComponent<RectTransform>();
        rectTransform.anchoredPosition = new Vector2(0, 0);

        TextMesh textMesh = label.AddComponent<TextMesh>();
        textMesh.text = constraints[i].name;
        textMesh.color = textColor;
        textMesh.anchor = TextAnchor.UpperCenter;

        return cube;
    }

    public void ChangeLOD() {
        DeactivateScenes();
        var value = cognitiveLoadSlider.value;

        if (UIGenerated == false) {
            Debug.Log("UI not generated.");
        } else {
            for (int i = 0; i < numPanels; i++) {
                GameObject go = panels[i];
                TextMesh textMesh = go.transform.GetChild(0).GetComponent<TextMesh>();
                if (value < 4) {
                    go.transform.localScale = new Vector3(panelSizes[i, 1], panelSizes[i, 0], 0.001f);
                    textMesh.text = constraints[i].name;
                } else if (value >= 4 && value < 7) {
                    go.transform.localScale = new Vector3(panelSizes[i, 1], panelSizes[i, 0], 0.001f);
                    textMesh.text = constraints[i].name.Substring(0, 1);
                 } else {
                    go.transform.localScale = new Vector3(0.05f, 0.05f, 0.05f);
                    textMesh.text = constraints[i].name.Substring(0, 1);
                }
            }

            if (value < 4) {
                officeScene.SetActive(true);
            } else if (value >= 4 && value < 7) {
                classScene.SetActive(true);
            } else {
                labScene.SetActive(true);
            }
        }
    }

    private void OnDestroy() {
        _pythonNetworking.StopClient();
    }
}