using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.UI;

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
    public string imageMetaFile;
    public float[,] panelSizes;
    public PanelConstraints[] constraints;

    private PythonNetworking pythonNetworking;
    private GameObject[] panels;
    private List<string> panelData;
    private List<string> prefLearningData;
    private int numPanels;
    private bool UIGenerated;
    private bool imageFileInput;
    private string defaultBufferFile;
    
    private GameObject instructionMenu;
    private GameObject labScene;
    private GameObject classScene;
    private GameObject officeScene;

    public void Start() {
        pythonNetworking = new PythonNetworking();
        numPanels = constraints.Length;
        panelSizes = new float[numPanels, 2];
        UIGenerated = false;
        imageFileInput = false;
        defaultBufferFile = "";

        if (imageBufferFile != "")
            imageFileInput = true;

        instructionMenu = GameObject.FindGameObjectWithTag("InstructionMenu");
        labScene = GameObject.FindGameObjectWithTag("labScene");
        classScene = GameObject.FindGameObjectWithTag("classScene");
        officeScene = GameObject.FindGameObjectWithTag("officeScene");

        Debug.Log("UI tool initialized!");
        DeactivateScenes();
    }

    private void DeactivateScenes() {
        labScene.SetActive(false);
        classScene.SetActive(false);
        officeScene.SetActive(false);
    }

    public void SubmitConstraints() {
        Debug.Log("Submitting constraints...");
        StartCoroutine(CreateRequest("P"));
    }

    public void StartPreferenceLearning() {
        if (panelData != null) {
            Debug.Log("Starting preference learning application...");
            StartCoroutine(CreateRequest("L"));
        } else {
            Debug.Log("Please submit constraints and press 'show optimal UI'.");
        }
    }

    public void CreateUI() {
        instructionMenu.SetActive(false);
        int i = 0;
        
        if (panelData != null) {
            Debug.Log("Creating UI!");
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

    private Serialization.Request _CreateRequest() {
        Debug.Log("Creating request...");
        Serialization.Request request = null;
        if (imageFileInput == true) {
            request = new Serialization.Request(imageBufferFile, imageMetaFile, numPanels, constraints, enableOcclusion.isOn, enableColorHarmony.isOn,
                                                colorfulnessSlider.value, edgenessSlider.value, fittsLawSlider.value,
                                                ceSlider.value, muscleActivationSlider.value, rulaSlider.value);
        } else {
            if (defaultBufferFile == "") {
                defaultBufferFile = "office_buff.log";
                officeScene.SetActive(true);
            }
            request = new Serialization.Request(defaultBufferFile, "meta.log", numPanels, constraints, enableOcclusion.isOn, enableColorHarmony.isOn,
                                                colorfulnessSlider.value, edgenessSlider.value, fittsLawSlider.value,
                                                ceSlider.value, muscleActivationSlider.value, rulaSlider.value);
        }
        return request;
    }
   
    private IEnumerator CreateRequest(string type) {
        Serialization.Request request = _CreateRequest();
        var requestJson = JsonUtility.ToJson(request);
        pythonNetworking.PerformRequest(type, requestJson);
        yield return new WaitUntil(() => pythonNetworking.requestResult != null);

        if (type == "P") {
            panelData = JsonConvert.DeserializeObject<List<string>>(pythonNetworking.requestResult);
        } else {
            prefLearningData = JsonConvert.DeserializeObject<List<string>>(pythonNetworking.requestResult);
            UpdatePanels(prefLearningData);
        }
    }

    private void UpdatePanels(List<string> data) {
        GameObject[] panels = GameObject.FindGameObjectsWithTag("Panel");
        int i = 0;

        foreach(GameObject p in panels) {
            string[] spl = data[i].Split(';');
            float[] panelPos = {float.Parse(spl[0].Split(',')[1]), float.Parse(spl[0].Split(',')[0]), float.Parse(spl[0].Split(',')[2])};
            Color panelColor = new Color(float.Parse(spl[1].Split(',')[0])/255, float.Parse(spl[1].Split(',')[1])/255, float.Parse(spl[1].Split(',')[2])/255);

            p.transform.position = new Vector3(panelPos[1], panelPos[0], 0.5f);
            var panelRenderer = p.GetComponent<Renderer>();
            panelRenderer.material.SetColor("_Color", panelColor);
            i++;
        }
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

        GameObject panel = GameObject.CreatePrimitive(PrimitiveType.Cube);
        panel.tag = "Panel";
    
        panel.transform.position = new Vector3(panelPos[1], panelPos[0], 0.5f);
        panel.transform.localScale = new Vector3(width, height, 0.0001f);
        var panelRenderer = panel.GetComponent<Renderer>();
        panelRenderer.material.SetColor("_Color", panelColor);

        GameObject label = new GameObject();
        label.transform.parent = panel.transform;
        label.transform.localPosition = panel.transform.localPosition;
        label.transform.localScale = new Vector3(panel.transform.localScale.y, panel.transform.localScale.x, 1.0f);

        RectTransform rectTransform = label.AddComponent<RectTransform>();
        rectTransform.anchoredPosition = new Vector2(0, 0);

        TextMesh textMesh = label.AddComponent<TextMesh>();
        textMesh.text = constraints[i].name;
        textMesh.color = textColor;
        textMesh.anchor = TextAnchor.UpperCenter;

        return panel;
    }

    public void ChangeLOD() {
        DeactivateScenes();
        var value = cognitiveLoadSlider.value;

        if (UIGenerated == false) {
            Debug.Log("UI not generated yet!");
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
                defaultBufferFile = "office_buff.log";
            } else if (value >= 4 && value < 7) {
                classScene.SetActive(true);
                defaultBufferFile = "classroom_buff.log";
            } else {
                labScene.SetActive(true);
                defaultBufferFile = "lab_buff.log";
            }
        }
    }

    private void OnDestroy() {
        pythonNetworking.StopClient();
    }
}