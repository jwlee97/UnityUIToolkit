using System.Collections.Generic;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;

public class PythonNetworking {
    private bool clientStopped;
    private RequestSocket requestSocket;

    private byte[] frame;
    // for now only one request at a time is supported
    public string requestResult;
    private bool isAvailable;

    public PythonNetworking() {
        clientStopped = false;
        var clientThread = new Thread(NetMQClient);
        clientThread.Start();
    }
    

    public void StopClient() {
        clientStopped = true;
    }

    // ReSharper disable once InconsistentNaming
    private void NetMQClient() {
        AsyncIO.ForceDotNet.Force();

        requestSocket = new RequestSocket();
        // "tcp://192.168.0.104:5555"
        requestSocket.Connect("tcp://127.0.0.1:5555");
        
        isAvailable = true;

        while (!clientStopped)
        {
            //Debug.Log("Continuing");
        }

        requestSocket.Close();
        NetMQConfig.Cleanup();
    }

    public void SetFrame(byte[] currFrame) {
        frame = currFrame;
    }

    // Create queue of requests in case multiple have to be handled
    private void SimpleRequest(string endpoint, string request) {
        // wait until socket is available
        while (!isAvailable) {
            //Debug.Log("Socket unavailable");
        }

        isAvailable = false;
        if (request == null) {
            requestSocket.SendFrame(endpoint);
        } else {
            Debug.Log("Sending to Python server: " + request);
            requestSocket.SendMoreFrame(endpoint);
            requestSocket.SendFrame(request);
        }

        var msg = requestSocket.ReceiveFrameBytes();
        isAvailable = true;
        requestResult = System.Text.Encoding.UTF8.GetString(msg);
    }

    public void PerformRequest(string endpoint, string request) {
        requestResult = null;
        var requestThread = new Thread(() => SimpleRequest(endpoint, request));
        requestThread.Start();
    }
}